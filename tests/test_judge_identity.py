from src.benchmark.leave_one_judge_out import select_votes

def test_substitute_stays_anthropic_when_in_google_slot():
    row={'panel':{'judge_3':{'provider':'anthropic','model_name':'claude-sonnet-5','answer':{'verdict':'yes'}}},'per_judge':{}}
    votes,_=select_votes(row)
    assert votes=={'anthropic:claude-sonnet-5':'yes'}
    assert 'google:gemini-3.5-flash' not in votes

def test_later_real_vote_preserves_real_identity_and_precedence():
    old={'provider':'openai','model_name':'gpt-5.5','answer':{'verdict':'yes'}}
    new={'provider':'openai','model_name':'gpt-5.5','answer':{'verdict':'no'}}
    votes,sources=select_votes({'panel':{'judge_1':old},'per_judge':{'openai__gpt-5.5':{'judge':{'provider':'openai','model_name':'gpt-5.5'},'record':new}}})
    assert votes['openai:gpt-5.5']=='no' and sources['openai:gpt-5.5'].startswith('per_judge/')
