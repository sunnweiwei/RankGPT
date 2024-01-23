from openai import OpenAI

# This file includes the implementation of Relevance Generation and Query Generation as described in the RankGPT paper. For more details, refer to the paper available at: https://arxiv.org/abs/2304.09542

client = OpenAI(api_key='sk-xxx')

FEW_SHOT_EXAMPLE = '''Given a passage and a question, predict whether the passage includes an answer to the question by producing either `Yes` or `No`.

Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: Yes

Passage: RE: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.
Query: how many eye drops per ml
Does the passage answer the query?
Answer: No

Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a Wells Fargo branch. 1 Money in — deposits.
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: No

Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many bank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking.
Query: can you open a wells fargo account online
Does the passage answer the query?
Answer: Yes
'''

ZERO_SHOT_EXAMPLE = '''Given a passage and a question, predict whether the passage includes an answer to the question by producing either `Yes` or `No`.'''


def relevance_generation(query, passage, instruction: str = ZERO_SHOT_EXAMPLE, model='gpt-3.5-turbo'):
    prompt = f"{instruction}\nPassage: {passage}\nQuery: {query}\nDoes the passage answer the query?\nAnswer:"
    if 'instruct' in model or 'text' in model or 'davinci' in model:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            temperature=0, logprobs=5, max_tokens=2
        )
        text = response.choices[0].text
        token_logprobs = response.choices[0].logprobs.token_logprobs[0]
        top_logprobs = response.choices[0].logprobs.top_logprobs[0]
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=2, logprobs=True, top_logprobs=5
        )
        text = response.choices[0].message.content
        token_logprobs = response.choices[0].logprobs.content[0].logprob
        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
        top_logprobs = {word.token: word.logprob for word in top_logprobs}

    if 'Yes' in text:
        logprobs = token_logprobs
        logprobs = - 1 / logprobs
        rel = logprobs
    elif 'No' in text:
        logprobs = token_logprobs
        logprobs = 1 / logprobs
        rel = logprobs
    else:
        if ' Yes' in top_logprobs and ' No' in top_logprobs and top_logprobs[' Yes'] > top_logprobs[' No']:
            logprobs = top_logprobs[' Yes']
            logprobs = - 1 / logprobs
            rel = logprobs
        elif ' Yes' in top_logprobs and ' No' in top_logprobs and top_logprobs[' Yes'] < top_logprobs[' No']:
            logprobs = top_logprobs[' No']
            logprobs = 1 / logprobs
            rel = logprobs
        elif ' Yes' in top_logprobs:
            logprobs = top_logprobs[' Yes']
            logprobs = - 1 / logprobs
            rel = logprobs
        elif ' No' in top_logprobs:
            logprobs = top_logprobs[' No']
            logprobs = 1 / logprobs
            rel = logprobs
        elif 'yes' in text.lower():
            rel = 0
        else:
            rel = -1000000
    return rel


def query_generation(query, passage, model='davinci-002'):
    prompt = [f"Please write a question based on this passage.\nPassage: {passage}\Question:",
              f" {query}"]

    response = client.completions.create(
        model=model,
        prompt=prompt[0] + prompt[1],
        temperature=0, logprobs=0, max_tokens=0, echo=True
    )
    # print(response)
    out = response.choices[0]
    assert prompt[0] + prompt[1] == out.text
    i = out.logprobs.text_offset.index(len(prompt[0]) - 1)
    if i == 0:
        i = i + 1
    loss = -sum(out.logprobs.token_logprobs[i:-1])  # ignore the last '.'
    avg_loss = loss / (len(out.logprobs.text_offset) - i - 1)  # 1 is the last '.'
    rel = avg_loss
    return rel


def main():
    query = 'hello world'
    passage1 = '''A "Hello, World!" program is generally a simple computer program which outputs (or displays) to the screen (often the console) a message similar to "Hello, World!" while ignoring any user input. A small piece of code in most general-purpose programming languages, this program is used to illustrate a language's basic syntax. A "Hello, World!" program is often the first written by a student of a new programming language,[1] but such a program can also be used as a sanity check to ensure that the computer software intended to compile or run source code is correctly installed, and that its operator understands how to use it.'''
    passage2 = '''Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.[31] Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming. It is often described as a "batteries included" language due to its comprehensive standard library.[32][33] Guido van Rossum began working on Python in the late 1980s as a successor to the ABC programming language and first released it in 1991 as Python 0.9.0.[34] Python 2.0 was released in 2000. Python 3.0, released in 2008, was a major revision not completely backward-compatible with earlier versions.'''

    print(relevance_generation(query, passage1, model='gpt-3.5-turbo'))
    print(relevance_generation(query, passage2, model='gpt-3.5-turbo'))

    print(query_generation(query, passage1, model='babbage-002'))
    print(query_generation(query, passage2, model='babbage-002'))


if __name__ == '__main__':
    main()
