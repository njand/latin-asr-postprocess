"""
1:1 Replica of CLTK v0 Latin Word Tokenizer with Origin Metadata Tracking.
Matches tokenization logic for Latin-BERT while retaining origin metadata 
for lossless Inverse Text Normalization (ITN) detokenization.
"""

from typing import List, Tuple, NamedTuple, Optional
from latin_itn.config import PUNCT_MAP

QUE_EXCEPTIONS = [
    'quisque', 'quidque', 'quicque', 'quodque', 'cuiusque', 'cuique', 'quemque', 'quamque', 'quoque',
    'quaque', 'quique', 'quaeque', 'quorumque', 'quarumque', 'quibusque', 'quosque', 'quasque',
    'uterque', 'utraque', 'utrumque', 'utriusque', 'utrique', 'utramque', 'utroque',
    'utraeque', 'utrorumque', 'utrarumque', 'utrisque', 'utrosque', 'utrasque',
    'quicumque', 'quidcumque', 'quodcumque', 'cuiuscumque', 'cuicumque', 'quemcumque', 'quamcumque',
    'quocumque', 'quacumque', 'quaecumque', 'quorumcumque', 'quarumcumque', 'quibuscumque',
    'quoscumque', 'quascumque',
    'unusquisque', 'unaquaeque', 'unumquodque', 'unumquidque', 'uniuscuiusque', 'unicuique',
    'unumquemque', 'unamquamque', 'unoquoque', 'unaquaque',
    'plerusque', 'pleraque', 'plerumque', 'plerique', 'pleraeque', 'pleroque', 'pleramque',
    'plerorumque', 'plerarumque', 'plerisque', 'plerosque', 'plerasque',
    'absque', 'abusque', 'adaeque', 'adusque', 'aeque', 'antique', 'atque', 'circumundique', 'conseque',
    'cumque', 'cunque', 'denique', 'deque', 'donique', 'hucusque', 'inique', 'inseque', 'itaque',
    'longinque', 'namque', 'neque', 'oblique', 'peraeque', 'praecoque', 'propinque', 'qualiscumque',
    'quandocumque', 'quandoque', 'quantuluscumque', 'quantumcumque', 'quantuscumque', 'quinque',
    'quomodocumque', 'quomque', 'quotacumque', 'quotcumque', 'quotienscumque',
    'quotiensque', 'quotusquisque', 'quousque', 'relinque', 'simulatque', 'torque', 'ubicumque',
    'ubique', 'undecumque', 'undique', 'usque', 'usquequaque', 'utcumque', 'utercumque', 'utique',
    'utrimque', 'utrobique', 'utrubique'
]

NE_EXCEPTIONS = [
    'absone', 'acharne', 'acrisione', 'acumine', 'adhucine', 'adsuetudine', 'aeetine', 'aeschynomene',
    'aesone', 'agamemnone', 'agmine', 'albane', 'alcyone', 'almone', 'alsine', 'amasene', 'ambitione',
    'amne', 'amoene', 'amymone', 'anadyomene', 'andrachne', 'anemone', 'aniene', 'anne', 'antigone',
    'aparine', 'apolline', 'aquilone', 'arachne', 'arne', 'arundine', 'ascanione', 'asiane', 'asine',
    'aspargine', 'babylone', 'barine', 'bellone', 'belone', 'bene', 'benigne', 'bipenne', 'bizone',
    'bone', 'bubone', 'bulbine', 'cacumine', 'caligine', 'calymne', 'cane', 'carcine', 'cardine',
    'carmine', 'catacecaumene', 'catone', 'cerne', 'certamine', 'chalbane', 'chamaedaphne',
    'chamaemyrsine', 'chaone', 'chione', 'christiane', 'clymene', 'cognomine', 'commagene', 'commune',
    'compone', 'concinne', 'condicione', 'condigne', 'cone', 'confine', 'consone', 'corone', 'crastine',
    'crepidine', 'crimine', 'crine', 'culmine', 'cupidine', 'cyane', 'cydne', 'cyllene', 'cyrene',
    'daphne', 'depone', 'desine', 'dicione', 'digne', 'dine', 'dione', 'discrimine', 'diutine', 'dracone',
    'dulcedine', 'elatine', 'elephantine', 'elleborine', 'epidamne', 'erigone', 'euadne', 'euphrone',
    'euphrosyne', 'examine', 'faune', 'femine', 'feminine', 'ferrugine', 'fine', 'flamine', 'flumine',
    'formidine', 'fragmine', 'fraterne', 'fulmine', 'fune', 'germane', 'germine', 'geryone', 'gorgone',
    'gramine', 'grandine', 'haecine', 'halcyone', 'hammone', 'harundine', 'hedone', 'helene', 'helxine',
    'hermione', 'heroine', 'hesione', 'hicine', 'hicne', 'hierabotane', 'hippocrene', 'hispane',
    'hodierne', 'homine', 'hominesne', 'hortamine', 'hucine', 'humane', 'hunccine', 'huncine', 'iasione',
    'iasone', 'igne', 'imagine', 'immane', 'immune', 'impoene', 'impone', 'importune', 'impune', 'inane',
    'inconcinne', 'indagine', 'indigne', 'inferne', 'inguine', 'inhumane', 'inpone', 'inpune', 'insane',
    'insigne', 'inurbane', 'ismene', 'istucine', 'itone', 'iuuene', 'karthagine', 'labiene',
    'lacedaemone', 'lanugine', 'latine', 'legione', 'lene', 'lenone', 'libidine', 'limine', 'limone',
    'lumine', 'magne', 'maligne', 'mane', 'margine', 'marone', 'masculine', 'matutine', 'medicamine',
    'melpomene', 'memnone', 'mesene', 'messene', 'misene', 'mitylene', 'mnemosyne', 'moderamine', 'moene',
    'mone', 'mortaline', 'mucrone', 'munimine', 'myrmidone', 'ne', 'necne', 'neptune', 'nequene',
    'nerine', 'nocturne', 'nomine', 'nonne', 'nullane', 'numine', 'nuncine', 'nyctimene', 'obscene',
    'obsidione', 'oenone', 'omine', 'omne', 'oppone', 'opportune', 'ordine', 'origine', 'orphne',
    'oxymyrsine', 'paene', 'pallene', 'pane', 'paraetacene', 'patalene', 'pectine', 'pelagine', 'pellene',
    'pene', 'perbene', 'perbenigne', 'peremne', 'perenne', 'perindigne', 'peropportune', 'persephone',
    'phryne', 'pirene', 'pitane', 'plane', 'pleione', 'plene', 'pone', 'praefiscine', 'prasiane',
    'priene', 'priuigne', 'procne', 'proditione', 'progne', 'prone', 'propone', 'pulmone', 'pylene',
    'pyrene', 'pythone', 'ratione', 'regione', 'religione', 'remane', 'retine', 'rhene', 'rhododaphne',
    'robigine', 'romane', 'roxane', 'rubigine', 'sabine', 'sane', 'sanguine', 'saturne', 'seditione',
    'segne', 'selene', 'semine', 'semiplene', 'sene', 'sepone', 'serene', 'sermone', 'serrane', 'siccine',
    'sicine', 'sine', 'sithone', 'solane', 'sollemne', 'somne', 'sophene', 'sperne', 'spiramine',
    'stamine', 'statione', 'stephane', 'sterne', 'stramine', 'subpone', 'subtegmine', 'subtemine',
    'sulmone', 'superne', 'supine', 'suppone', 'susiane', 'syene', 'tantane', 'tantine', 'taprobane',
    'tegmine', 'telamone', 'temne', 'temone', 'tene', 'testudine', 'theophane', 'therone', 'thyone',
    'tiberine', 'tibicine', 'tiburne', 'tirone', 'tisiphone', 'torone', 'transitione', 'troiane',
    'turbine', 'turne', 'tyrrhene', 'uane', 'uelamine', 'uertigine', 'uesane', 'uimine', 'uirgine',
    'umbone', 'unguine', 'uolumine', 'uoragine', 'urbane', 'uulcane', 'zone'
]

N_EXCEPTIONS = [
    'aenean', 'agmen', 'alioquin', 'an', 'attamen', 'cacumen', 'carmen', 'certamen', 'clymenen', 'cognomen',
    'crimen', 'culmen', 'dein', 'deucalion', 'discrimen', 'en', 'epitheton', 'erinyn', 'exin', 'flumen',
    'forsan', 'forsitan', 'fulmen', 'gramen', 'hymen', 'iason', 'in', 'limen', 'liquamen', 'lumen', 'nomen',
    'non', 'numen', 'omen', 'orion', 'paean', 'pan', 'pelion', 'phaethon', 'python', 'quin', 'semen', 'sin',
    'specimen', 'tamen', 'themin', 'titan', 'alcuin', 'caen', 'christian', 'chronicon', 'châtillon', 'claudian',
    'john', 'justin', 'latin', 'lucan', 'martin', 'novatian', 'quintilian', 'roman', 'tertullian'
]

UE_EXCEPTIONS = [
    'agaue', 'ambigue', 'assidue', 'aue', 'boue', 'breue', 'calue', 'caue', 'ciue', 'congrue', 'contigue',
    'continue', 'curue', 'exigue', 'exue', 'fatue', 'faue', 'fue', 'furtiue', 'gradiue', 'graue',
    'ignaue', 'incongrue', 'ingenue', 'innocue', 'ioue', 'lasciue', 'leue', 'moue', 'mutue', 'naue',
    'neue', 'niue', 'perexigue', 'perspicue', 'pingue', 'praecipue', 'praegraue', 'prospicue', 'proterue',
    'remoue', 'resolue', 'saeue', 'salue', 'siue', 'solue', 'strenue', 'sue', 'summoue', 'superflue',
    'supplicue', 'tenue', 'uiue', 'ungue', 'uoue'
]

VE_EXCEPTIONS = [
    'agave', 'ave', 'bove', 'breve', 'calve', 'cave', 'cive', 'curve', 'fave', 'furtive', 'gradive',
    'grave', 'ignave', 'iove', 'lascive', 'leve', 'move', 'nave', 'neve', 'nive', 'praegrave',
    'promiscue', 'prospicve', 'proterve', 'remove', 'resolve', 'saeve', 'salve', 'sive', 'solve',
    'summove', 'vive', 'vove'
]

ST_EXCEPTIONS = [
    'abest', 'adest', 'ast', 'deest', 'est', 'inest', 'interest', 'post', 'potest', 'prodest', 'subest', 'superest'
]

ENCLITICS = ['que', 'n', 'ne', 'ue', 've', 'st']

LATIN_EXCEPTIONS = set(
    QUE_EXCEPTIONS + NE_EXCEPTIONS + N_EXCEPTIONS + UE_EXCEPTIONS + VE_EXCEPTIONS + ST_EXCEPTIONS + ENCLITICS
)

# Word replacements mapped directly per token to preserve original word boundaries
LATIN_REPLACEMENTS_MAP = {
    'mecum': ('cum', 'me'),
    'tecum': ('cum', 'te'),
    'secum': ('cum', 'se'),
    'nobiscum': ('cum', 'nobis'),
    'vobiscum': ('cum', 'vobis'),
    'quocum': ('cum', 'quo'),
    'quacum': ('cum', 'qua'),
    'quicum': ('cum', 'qui'),
    'quibuscum': ('cum', 'quibus'),
    'sodes': ('si', 'audes'),
    'satin': ('satis', 'ne'),
    'scin': ('scis', 'ne'),
    'sultis': ('si', 'vultis'),
    'similist': ('similis', 'est'),
    'qualist': ('qualis', 'est')
}


class TokenInfo(NamedTuple):
    """Container for token string, enclitic flag, and original pre-tokenized word stem."""
    token_str: str
    is_enclitic: bool
    orig_word: Optional[str] = None


class CLTKLegacyLatinTokenizer:
    """CLTK v0 word tokenizer with origin metadata tracking for ITN detokenization."""

    def __init__(self):
        self.enclitics = ENCLITICS
        self.exceptions = LATIN_EXCEPTIONS
        self.replacements_map = LATIN_REPLACEMENTS_MAP

    def tokenize(self, text: str) -> List[TokenInfo]:
        """
        Tokenizes string into tokens and tracks origin metadata.
        Returns: [TokenInfo(token_str, is_enclitic, orig_word), ...]
        """
        raw_tokens = text.strip().split()
        final_tokens: List[TokenInfo] = []

        for token in raw_tokens:
            token_lower = token.lower()

            # 1. Check for compound word replacements (e.g. "mecum" -> "cum" + "me")
            if token_lower in self.replacements_map:
                head, tail = self.replacements_map[token_lower]
                final_tokens.append(TokenInfo(token_str=head, is_enclitic=False, orig_word=token))
                final_tokens.append(TokenInfo(token_str=tail, is_enclitic=True, orig_word=token))
                continue

            # 2. Check for enclitics if not in exceptions
            is_enclitic = False
            if token_lower not in self.exceptions:
                for enclitic in self.enclitics:
                    if token_lower.endswith(enclitic):
                        if enclitic == 'n':
                            stem = token[:-1]
                            final_tokens.append(TokenInfo(token_str=stem, is_enclitic=False, orig_word=token))
                            final_tokens.append(TokenInfo(token_str='-ne', is_enclitic=True, orig_word=token))
                        elif enclitic == 'st':
                            stem = token[:-1] if token_lower.endswith('ust') else token[:-2]
                            final_tokens.append(TokenInfo(token_str=stem, is_enclitic=False, orig_word=token))
                            final_tokens.append(TokenInfo(token_str='est', is_enclitic=True, orig_word=token))
                        else:
                            stem = token[:-len(enclitic)]
                            enc_str = f"-{token[-len(enclitic):].lower()}"
                            final_tokens.append(TokenInfo(token_str=stem, is_enclitic=False, orig_word=None))
                            final_tokens.append(TokenInfo(token_str=enc_str, is_enclitic=True, orig_word=None))
                        
                        is_enclitic = True
                        break

            if not is_enclitic:
                final_tokens.append(TokenInfo(token_str=token, is_enclitic=False, orig_word=None))

        return final_tokens


def rejoin_cltk_enclitics_and_format(
    tokens_with_metadata: List[Tuple[str, bool] | TokenInfo], 
    predicted_tags: List[str]
) -> str:
    """
    Rejoins enclitics and expanded compound forms back to host words
    and applies predicted ITN casing and punctuation.
    """
    formatted_words = []

    for meta, tag in zip(tokens_with_metadata, predicted_tags, strict=True):
        token_str = meta[0]
        is_enclitic = meta[1]
        orig_word = getattr(meta, "orig_word", None)
        if orig_word is None and len(meta) > 2:
            orig_word = meta[2]

        parts = tag.split("_")
        casing = parts[0]
        punct_type = parts[1] if len(parts) > 1 else "NONE"
        punct = PUNCT_MAP.get(punct_type, "")

        clean_token = token_str[1:] if token_str.startswith("-") else token_str

        if is_enclitic and formatted_words:
            prev_word = formatted_words.pop()
            prev_word_clean = prev_word.rstrip(".,?!;: ")

            if orig_word:
                # Reconstruct expanded original form ("vin", "mecum", "satin") using host casing
                is_head_capitalized = prev_word_clean[0].isupper() if prev_word_clean else (casing == "TITLE")
                base_word = orig_word.capitalize() if is_head_capitalized else orig_word.lower()
                merged = base_word + punct
            else:
                # Standard attached enclitics ("-que", "-ve")
                word_formatted = clean_token.capitalize() if casing == "TITLE" else clean_token.lower()
                merged = prev_word_clean + word_formatted + punct

            formatted_words.append(merged)
        else:
            word_formatted = clean_token.capitalize() if casing == "TITLE" else clean_token.lower()
            formatted_words.append(word_formatted + punct)

    return " ".join(formatted_words)