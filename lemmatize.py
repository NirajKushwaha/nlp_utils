### =====================================================================
### TOKENIZER W/ LEMMATIZATION FOR BADLY CASED TEXT (e.g. social media)
### =====================================================================

## usage

## pip install spacy truecase
## python -m spacy download en_core_web_sm

## single text
# text = ['text is this']
# text_lemmatized = lemmatize(text)

## multiple texts
## texts = ['text1 is this','text2 is that']
## texts_lemmatized = lemmatize_many(texts,batch_size=10,n_process=1)

import spacy
import truecase
from spacy.util import filter_spans
from tqdm import tqdm

def fix_hyphens_in_tokenizer(nlp):
    from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER, CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
    from spacy.util import compile_infix_regex

    ## https://stackoverflow.com/a/58112065
    infixes = (
        LIST_ELLIPSES
        + LIST_ICONS
        + [
            r"(?<=[0-9])[+\-\*^](?=[0-9-])",
            r"(?<=[{al}{q}])\.(?=[{au}{q}])".format(
                al=ALPHA_LOWER, au=ALPHA_UPPER, q=CONCAT_QUOTES
            ),
            r"(?<=[{a}]),(?=[{a}])".format(a=ALPHA),
            #r"(?<=[{a}])(?:{h})(?=[{a}])".format(a=ALPHA, h=HYPHENS),
            r"(?<=[{a}0-9])[:<>=/](?=[{a}])".format(a=ALPHA),
        ]
    )
    infix_re = compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_re.finditer
    return nlp

nlp = spacy.load("en_core_web_sm")
nlp = fix_hyphens_in_tokenizer(nlp)

try:
    import truecase
    _HAS_TRUECASE = True
except ImportError:
    _HAS_TRUECASE = False

_ARTICLES = {"the", "a", "an"}

# stopwords that legitimately appear *inside* a name; anything else
# marks the span as a mis-detection.
_NAME_CONNECTORS = {"of", "and", "the", "a", "an", "de", "van", "von",
                    "der", "den", "&", "'s"}



def _looks_miscased(text):
    """Capitalization is unreliable (Title Case / ALL CAPS), which wrecks
    en_core_web_sm's NER and POS tagging."""
    alpha = [w for w in text.split() if w[:1].isalpha()]
    if not alpha:
        return False
    return sum(w[0].isupper() for w in alpha) / len(alpha) > 0.6

# numeric-family entities: don't merge these into chunks — let their
# tokens fall through to the strip filter below.
_SKIP_ENT_LABELS = {"MONEY", "PERCENT", "QUANTITY", "CARDINAL",
                    "ORDINAL", "DATE", "TIME"}


def _entity_spans(doc):
    """Entity spans: leading articles trimmed, spurious stopword-laden spans
    dropped, and numeric-family labels skipped so numbers/currency can be
    stripped rather than glued into a chunk."""
    for ent in doc.ents:
        if ent.label_ in _SKIP_ENT_LABELS:
            continue
        start = ent.start
        while start < ent.end and doc[start].lower_ in _ARTICLES:
            start += 1
        if start >= ent.end:
            continue
        span = doc[start:ent.end]
        if any(t.is_stop and t.lower_ not in _NAME_CONNECTORS for t in span):
            continue
        yield span

def _compound_spans(doc):
    """Maximal runs joined by the 'compound' dependency, e.g. 'climate change'."""
    i, n = 0, len(doc)
    while i < n:
        j = i
        while j + 1 < n and doc[j].dep_ == "compound" and doc[j].head.i == j + 1:
            j += 1
        if j > i:
            yield doc[i:j + 1]
            i = j + 1
        else:
            i += 1       
            
def _process_doc(doc):
    """Everything that happens AFTER nlp() — takes a Doc, returns tokens."""
    spans = list(_entity_spans(doc)) + list(_compound_spans(doc))
    spans = filter_spans(spans)

    with doc.retokenize() as retok:
        for span in spans:
            retok.merge(span)

    out = []
    for tok in doc:
        if tok.is_punct or tok.is_space or tok.is_stop:
            continue
        if " " in tok.text:
            out.append(tok.text.lower().replace(" ", "_"))
            continue
        if tok.like_num or tok.is_currency or not any(c.isalpha() for c in tok.text):
            continue
        out.append(tok.lemma_.lower())
    return out


def _prep(text):
    """The truecasing step, which runs outside spaCy."""
    if _HAS_TRUECASE and _looks_miscased(text):
        return truecase.get_true_case(text)
    return text


def lemmatize(text):
    """Single-string convenience wrapper."""
    return _process_doc(nlp(_prep(text)))


def lemmatize_many(texts, 
                   batch_size=200, 
                   n_process=50, 
                   progress=True):
    print("Fixing casing...")
    prepped = [_prep(t) for t in texts]
    print("Tokenizing...")
    docs = nlp.pipe(prepped, batch_size=batch_size, n_process=n_process)
    if progress:
        docs = tqdm(docs, total=len(prepped), desc="lemmatizing")
    return [_process_doc(doc) for doc in docs]


