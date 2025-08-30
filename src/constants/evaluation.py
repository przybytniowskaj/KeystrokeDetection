PROMPT_ALPHANUM = """
You are given a list of keypresses predicted by a model that represent some meaning words. Each keypress corresponds to one element in the input array. The list can contain alphanumeric characters.

Although the model predicts most keypresses correctly, there may be occasional errors (wrong letters or numbers). Your taskis to reconstruct the original words, based on the given predictions.

Important requirements:
- Fix obvious typing mistakes so the output is a coherent, grammatically correct English text, but without changing the order of the characters.
- The total number of input keypresses must exactly match the total number of keypresses required to produce the fixed output.
- If you think a key is mistyped, you may replace it inplace with the correct one. However you cannot delete it or add new ones.

Example:
Input: [ ['h', 'a', 'l', 'k', 'o'], ['w', 'o', 'r', 'l', 'e'] ]
Expected output: [ ['h', 'e', 'l', 'l', 'o' ], ['w', 'o', 'r', 'l', 'd'] ]
Explanation: 'a' was mismatched with 'e', 'k' was mistaken with 'l' and 'e' was misclassified with 'd'

Process following list of predictions: {input_data}

Return a list of key lists that are most probable, without any extra explanations, code, additional styling, or comments. You MUST NOT include any comments.
"""


PROMPT_ALL = """
You are given a list of keypresses predicted by a model that represent a sentence. Each keypress corresponds to one element in the input array. The list can contain alphanumeric characters and special keys (such as shifts, cmds, alts etc).

Although the model predicts most keypresses correctly, there may be occasional errors (wrong letters, numbers, or special keys). Your task is to reconstruct the original sentence, based on the given predictions.

Important requirements:
- Fix obvious typing mistakes so the output is a coherent, grammatically correct English sentence, but without changing the order of the characters.
- The total number of input keypresses must exactly match the total number of keypresses required to produce the fixed output.
- If you think a key is mistyped, you may replace it in place with the correct one. However, you cannot delete it or add new ones.

Example:
Input: ['caps', 'h', 'caps', 'a', 'l', 'l', 'o', 'space', 'b', 'e', 'u', 't', 'k', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
Expected output: ['caps', 'h', 'caps', 'e', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'i', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
Explanation: 'k' was mismatched with 'i' and 'e' was misclassified with 'd', two 'caps' keys make sense as they result in capitalized 'h' and the last keys create '!'

Process the following list of predictions: {input_data}

Return a list of keypresses that are most probable and create a meaningful sentence, provide it without any extra explanations, code, additional styling, or comments.
"""


TARGET_SENTENCES = {
    "Artificial Intelligence has been part of our world for 75 years now.": [
        "caps",
        "a",
        "caps",
        "r",
        "t",
        "i",
        "f",
        "i",
        "c",
        "i",
        "a",
        "l",
        "space",
        "lshift",
        "i",
        "n",
        "t",
        "e",
        "l",
        "l",
        "i",
        "g",
        "e",
        "n",
        "c",
        "e",
        "space",
        "h",
        "a",
        "s",
        "space",
        "b",
        "e",
        "e",
        "n",
        "space",
        "p",
        "a",
        "r",
        "t",
        "space",
        "o",
        "f",
        "space",
        "o",
        "u",
        "r",
        "space",
        "w",
        "o",
        "r",
        "l",
        "d",
        "space",
        "f",
        "o",
        "r",
        "space",
        "7",
        "5",
        "space",
        "y",
        "e",
        "a",
        "r",
        "s",
        "space",
        "n",
        "o",
        "w",
        "dot",
    ],
    "In order to succeed, we must first believe that we can.": [
        "lshift",
        "i",
        "n",
        "space",
        "o",
        "r",
        "d",
        "e",
        "r",
        "space",
        "t",
        "o",
        "space",
        "s",
        "u",
        "c",
        "c",
        "e",
        "e",
        "d",
        "comma",
        "space",
        "w",
        "e",
        "space",
        "m",
        "u",
        "s",
        "t",
        "space",
        "f",
        "i",
        "r",
        "s",
        "t",
        "space",
        "b",
        "e",
        "l",
        "i",
        "e",
        "v",
        "e",
        "space",
        "t",
        "h",
        "a",
        "t",
        "space",
        "w",
        "e",
        "space",
        "c",
        "a",
        "n",
        "dot",
    ],
    "Life is a journey, not a destination.": [
        "rshift",
        "l",
        "i",
        "f",
        "e",
        "space",
        "i",
        "s",
        "space",
        "a",
        "space",
        "j",
        "o",
        "u",
        "r",
        "n",
        "e",
        "y",
        "comma",
        "space",
        "n",
        "o",
        "t",
        "space",
        "a",
        "space",
        "d",
        "e",
        "s",
        "t",
        "i",
        "n",
        "a",
        "t",
        "i",
        "o",
        "n",
        "dot",
    ],

}


TARGET_WORDS = {
    "Artificial Intelligence has been part of our world for 75 years now.": [
        ["a", "r", "t", "i", "f", "i", "c", "i", "a", "l"],
        ["i", "n", "t", "e", "l", "l", "i", "g", "e", "n", "c", "e"],
        ["h", "a", "s"],
        ["b", "e", "e", "n"],
        ["p", "a", "r", "t"],
        ["o", "f"],
        ["o", "u", "r"],
        ["w", "o", "r", "l", "d"],
        ["f", "o", "r"],
        ["7", "5"],
        ["y", "e", "a", "r", "s"],
        ["n", "o", "w"],
    ],
    "In order to succeed, we must first believe that we can.": [
        ["i", "n"],
        ["o", "r", "d", "e", "r"],
        ["t", "o"],
        ["s", "u", "c", "c", "e", "e", "d"],
        ["w", "e"],
        ["m", "u", "s", "t"],
        ["f", "i", "r", "s", "t"],
        ["b", "e", "l", "i", "e", "v", "e"],
        ["t", "h", "a", "t"],
        ["w", "e"],
        ["c", "a", "n"],
    ],
    "Life is a journey, not a destination.": [
        ["l", "i", "f", "e"],
        ["i", "s"],
        ["a"],
        ["j", "o", "u", "r", "n", "e", "y"],
        ["n", "o", "t"],
        ["a"],
        ["d", "e", "s", "t", "i", "n", "a", "t", "i", "o", "n"],
    ],
}
