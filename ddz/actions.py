"""Complete, factorized move space. Wings are chosen in sorted rank order.

Every server-legal interpretation has one body and one canonical wing sequence.
No hint pruning, action-count cap, or sampling of candidate moves is used.
"""
from dataclasses import dataclass
import numpy as np

TYPES = ('pass', 'single', 'pair', 'trio', 'trio1', 'trio2', 'straight',
         'pairs', 'plane', 'plane1', 'plane2', 'four2', 'four22', 'bomb', 'rocket')

@dataclass(frozen=True)
class Body:
    type: str
    rank: int
    length: int
    cards: tuple
    wing_unit: int = 0
    wings: int = 0

    @property
    def choice(self):
        return f'{self.type}:{self.rank}:{self.length}'


def catalogue():
    out = [Body('pass', 0, 1, (0,) * 15)]
    def add(t, r, n, unit, wunit=0, wings=0):
        c = [0] * 15
        for i in range(r - n + 1, r + 1):
            c[i] = unit
        out.append(Body(t, r, n, tuple(c), wunit, wings))
    for r in range(15):
        add('single', r, 1, 1)
    for r in range(13):
        for t, unit, wu, wn in [('pair',2,0,0),('trio',3,0,0),
                               ('trio1',3,1,1),('trio2',3,2,1),
                               ('four2',4,1,2),('four22',4,2,2),('bomb',4,0,0)]:
            add(t,r,1,unit,wu,wn)
    out.append(Body('rocket',14,1,(0,)*13+(1,1)))
    for t, unit, lo, hi, wu in [('straight',1,5,12,0), ('pairs',2,3,10,0),
                               ('plane',3,2,6,0), ('plane1',3,2,5,1),
                               ('plane2',3,2,4,2)]:
        for n in range(lo,hi+1):
            for r in range(n-1,12):
                add(t,r,n,unit,wu,n if wu else 0)
    return tuple(out)

BODIES = catalogue()
N_BODY = len(BODIES)
BID_OFFSET = N_BODY
WING_OFFSET = N_BODY + 4
N_ACTIONS = WING_OFFSET + 15
COUNTS = np.array([b.cards for b in BODIES], np.int32)
TYPE = np.array([TYPES.index(b.type) for b in BODIES], np.int32)
RANK = np.array([b.rank for b in BODIES], np.int32)
LENGTH = np.array([b.length for b in BODIES], np.int32)
WUNIT = np.array([b.wing_unit for b in BODIES], np.int32)
WINGS = np.array([b.wings for b in BODIES], np.int32)

def card_rank(card):
    if type(card) is not int or not 0 <= card < 54:
        raise ValueError('card id must be an integer in 0..53')
    return card // 4 if card < 52 else card - 39

def counts_of(cards):
    if len(set(cards)) != len(cards):
        raise ValueError('duplicate card ids')
    return np.bincount([card_rank(c) for c in cards], minlength=15).astype(np.int32)

def physical_cards(hand, counts, preferred=()):
    """Map rank counts to real cards, using revealed bottom IDs first."""
    need = list(map(int, counts))
    result = []
    revealed=set(preferred)
    for c in sorted(hand,key=lambda card:(card not in revealed,card)):
        r = card_rank(c)
        if need[r]:
            result.append(c)
            need[r] -= 1
    if any(need):
        raise ValueError('move contains cards outside hand')
    return sorted(result)

def interpretations(counts):
    """CPU move validation, including all ambiguous airplane interpretations."""
    c = np.asarray(counts)
    out = []
    for i,b in enumerate(BODIES[1:],1):
        rem = c - COUNTS[i]
        if np.any(rem < 0) or np.any((COUNTS[i] > 0) & (rem > 0)):
            continue
        if b.wing_unit == 0:
            ok = not np.any(rem)
        elif b.wing_unit == 1:
            ok = rem.sum() == b.wings and rem.max() <= 3 and not (rem[13] and rem[14])
        else:
            ok = np.all((rem == 0) | (rem == 2)) and rem.sum() == 2*b.wings
        if ok:
            out.append(i)
    return out

def beats(a,b):
    a,b = BODIES[a], BODIES[b]
    if b.type == 'pass': return a.type != 'pass'
    if a.type == 'rocket': return b.type != 'rocket'
    if a.type == 'bomb' and b.type not in ('bomb','rocket'): return True
    return a.type == b.type and a.length == b.length and a.rank > b.rank
