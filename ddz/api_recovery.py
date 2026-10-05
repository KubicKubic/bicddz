"""Legal degraded decisions for a state the model adapter cannot reconstruct."""
from .api import RetryDecision
from .actions import counts_of, physical_cards
import time


class GuardedPolicy:
    """Keep model failures and unchanged-version rejections inside the live loop."""
    def __init__(self,model,client,on_decision=None):
        self.model=model; self.client=client; self.on_decision=on_decision
        self.rejected_key=None

    def on_rejected(self,raw,endpoint,payload,error):
        self.rejected_key=(raw.get('id'),raw['version'])

    def decide(self,raw):
        started=time.monotonic(); reason=None; source='model'
        key=(raw.get('id'),raw['version'])
        try:
            if key==self.rejected_key: raise ValueError('Previous action rejected at this version')
            result=self.model.decide(raw)
        except (ValueError,KeyError,IndexError,TypeError,OverflowError) as error:
            source='fallback'; reason=f'{type(error).__name__}: {error}'
            result=safe_decision(raw,self.client)
        if self.on_decision:
            self.on_decision(raw,*result,source,time.monotonic()-started,reason)
        return result


def safe_decision(raw, client):
    """Preserve the version guard; refresh if leading hints are unavailable."""
    if raw['turn'] != raw['seat']:
        raise RetryDecision('No longer our turn')
    version = int(raw['version'])
    if raw['phase'] == 'bidding':
        value = int(raw['bid']) + 1 if raw.get('must_bid', False) else 0
        if value > 3:
            raise RetryDecision('Auction already closed')
        return 'bid', {'version': version, 'value': value}
    if raw['phase'] != 'playing':
        raise RetryDecision('No active decision')
    if not isinstance(raw.get('hand'),list):
        raise RetryDecision('Own hand missing; refresh full state')
    if not raw.get('leading', False):
        return 'pass', {'version': version}
    code, result = client.request(f"/games/{raw['id']}/hints")
    if code != 200 or not result.get('hints'):
        raise RetryDecision('Leading hints unavailable; refresh state')
    try:
        cards = physical_cards(raw['hand'], counts_of(result['hints'][0]), raw.get('bottom') or ())
        if not cards: raise ValueError('Empty leading hint')
    except (ValueError,KeyError,TypeError,IndexError) as error:
        raise RetryDecision('Hints no longer match this hand; refresh state') from error
    return 'play', {'version': version, 'cards': cards}
