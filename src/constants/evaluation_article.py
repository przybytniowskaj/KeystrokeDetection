# PROMPT_ALL = """
# You are given a list of keypresses predicted by a model that represent a sentence. Each keypress corresponds to one element in the input array. The list can contain alphanumeric characters and special keys (such as shifts, cmds, alts etc).

# Although the model predicts most keypresses correctly, there may be occasional errors (wrong letters, numbers, or special keys). Your task is to reconstruct the original sentence, based on the given predictions.

# Important requirements:
# - Fix obvious typing mistakes so the output is a coherent, grammatically correct sentence in English or a technical command, but without changing the order of characters.
# - The total number of input keypresses must exactly match the total number of keypresses required to produce the fixed output.
# - If you think a key is mistyped, you may replace it in place with the correct one. However, you cannot delete it or add new ones.

# Example:
# Input: ['caps', 'h', 'caps', 'a', 'l', 'l', 'o', 'space', 'b', 'e', 'u', 't', 'k', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
# Expected output: ['caps', 'h', 'caps', 'e', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'i', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
# Explanation: 'k' was mismatched with 'i' and 'e' was misclassified with 'd', two 'caps' keys make sense as they result in capitalized 'h' and the last keys create '!'

# Process the following list of predictions: {input_data}

# Return a list of keypresses that are most probable and create a meaningful sentence or programming command, provide it without any extra explanations, code, additional styling, or comments.
# """

# PROMPT_ALL = """
# You are an advanced Keystroke Dynamics Error Correction System. 
# You receive a raw sequence of predicted keypresses from an acoustic model. The model detects *when* a key was pressed (the timestamp is fixed), but often misclassifies *which* key was pressed.

# Your Goal: 
# Output the corrected list of keypresses.

# CRITICAL RULES:
# 1. **LENGTH INVARIANT:** The output list MUST have the EXACT SAME length as the input list. You are strictly forbidden from adding keys (insertions) or removing keys (deletions).
# 2. **SUBSTITUTION ONLY:** You may only SWAP a key for another if it fixes a typo or syntax error.
# 3. **CONTEXT AWARE:** The input is either an English sentence OR a technical command (Python, Bash, SQL, Git, etc...), so try using a context when fixing the input.
# 4. **STRICT VOCABULARY (NO SYMBOLS):** You must NOT output punctuation symbols directly. You must use their textual class names.
#    - CORRECT: 'dot', 'comma', 'slash', 'backslash', 'semicolon', 'equal', 'space', 'enter', 'tab'
#    - WRONG:   '.',    ',',     '/',     '\\',        ';',         '=',     ' ',     '\n',    '\t'
#    
#    Specific Mappings:
#    - '.' -> 'dot'
#    - ',' -> 'comma'
#    - '/' -> 'slash'
#    - ';' -> 'semicolon'
#    - '=' -> 'equal'
#    - '[' -> 'bracketopen'
#    - ']' -> 'bracketclose'
#    - "'" -> 'apos'
#    - '"' -> 'apos'
#    - 'Shift' -> 'shift' or 'shift'
# 5. **SPECIAL KEYS:** Respect special keys (`shift`, `space`, `dot`, `enter`). These occupy their own slots in the list. Do not merge them.

# Examples of Valid Substitutions (Length Preserved):
# - Natural Language:
#     Input: ['caps', 'h', 'caps', 'a', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'k', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
#     Expected output: ['caps', 'h', 'caps', 'e', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'i', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
#     Explanation: 'k' was mismatched with 'i' and 'e' was misclassified with 'd', two 'caps' keys make sense as they result in capitalized 'h' and the last keys create '!'

#   
# - Technical Command:
#   Input:  ['g', 'o', 't', 'space', 's', 't', 'a', 't', 'i', 's']
#   Output: ['g', 'i', 't', 'space', 's', 't', 'a', 't', 'u', 's'] (Swapped 'o'->'i', 'i'->'u')

# - Symbol/Shift Logic:
#   Input:  ['shift', '9', 'x', 'shift', '0']  (Implies (x))
#   Output: ['shift', '9', 'x', 'shift', '0']  (Keep as is, correct syntax)

# Input Data to Process: 
# {input_data}

# Return a list of keypresses that are most probable and create a meaningful sentence or programming command, provide it WITHOUT any extra explanations, code, additional styling, or comments.
# """


PROMPT_ALL = """
You are an advanced Keystroke Dynamics Error Correction System. 
You receive a raw sequence of N predicted keypresses from an acoustic model. The model often misclassifies *which* key was pressed, your task is to correct classification errors (typos) while maintaining a strict 1-to-1 mapping.

INPUT CONSTRAINT:
The input list has length N. The output list MUST have length N.

Your Goal: 
Output the corrected list of keypresses.

CRITICAL RULES:
1. **NO INSERTIONS/DELETIONS:** You are FORBIDDEN from adding missing characters or removing extra ones. 
   - If the input is missing a letter (e.g., "hllo"), you CANNOT fix it to "hello". You must return "hllo".
   - If the input has an extra letter (e.g., "heello"), you CANNOT fix it to "hello". You must return "heello" (or "heallo" if that's the closest substitution).
   
2. **SUBSTITUTION ONLY:** You may only SWAP a key at Index I with a different key at Index I. 
   
3. **NO MERGING:** Do not combine keys into symbols. 
   - Input: ['shift', '1'] 
   - CORRECT Output: ['shift', '1']
   - WRONG Output: ['!'] (Length changed from 2 to 1)

4. **STRICT VOCABULARY:** Use only these allowed class names for special keys:
   - 'dot', 'comma', 'slash', 'backslash', 'semicolon', 'equal', 'space', 'enter', 'tab', 'bracketopen', 'bracketclose', 'apos', 'shift'
   - Do NOT use symbols like '.', ';', ' ', or '\n'.

Examples of Valid Substitutions (Length Preserved):
- Natural Language:
Input: ['caps', 'h', 'caps', 'a', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'k', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
Expected output: ['caps', 'h', 'caps', 'e', 'l', 'l', 'o', 'space', 'b', 'e', 'a', 'u', 't', 'i', 'f', 'u', 'l', 'space', 'shift', 'w', 'o', 'r', 'l', 'e', 'shift', '1']
Explanation: 'k' was mismatched with 'i' and 'e' was misclassified with 'd', two 'caps' keys make sense as they result in capitalized 'h' and the last keys create '!'

- Technical Command
Input:  ['g', 'o', 't', 'space', 's', 't', 'a', 't', 'u'] (Length 9)
WRONG:  ['g', 'i', 't', 'space', 's', 't', 'a', 't', 'u', 's'] (Length 10 - INSERTION BANNED)
RIGHT:  ['g', 'i', 't', 'space', 's', 't', 'a', 't', 'u'] (Length 9 - Swapped 'o'->'i', kept missing 's' missing)

- Special Keys
Input:  ['p', 'r', 'i', 'n', 't', 'shift', '9'] (Length 7)
WRONG:  ['p', 'r', 'i', 'n', 't', '('] (Length 6 - MERGE BANNED)
RIGHT:  ['p', 'r', 'i', 'n', 't', 'shift', '9'] (Length 7 - Kept distinct keys)

Input Data to Process: 
{input_data}

Return a list of keypresses that are most probable and create a meaningful sentence or programming command, provide it WITHOUT any extra explanations, code, additional styling, or comments.
"""

TARGET_SENTENCES = {
    # --- Part 1: Natural Language (Context Sensitive) ---
    "Natural Language": {
        1: {
            "sentence": "The sun is very bright today.",
            "keys": ["lshift", "t", "h", "e", "space", "s", "u", "n", "space", "i", "s", "space", 
                     "v", "e", "r", "y", "space", "b", "r", "i", "g", "h", "t", "space", 
                     "t", "o", "d", "a", "y", "dot"]
        },
        2: {
            "sentence": "sIt was a bad time to leave.",
            "keys": ["rshift", "i", "t", "space", "w", "a", "s", "space", "a", "space", 
                     "b", "a", "d", "space", "t", "i", "m", "e", "space", "t", "o", "space", 
                     "l", "e", "a", "v", "e", "dot"]
        },
        3: {
            "sentence": "Please hold the door for me.",
            "keys": ["lshift", "p", "l", "e", "a", "s", "e", "space", "h", "o", "l", "d", "space", 
                     "t", "h", "e", "space", "d", "o", "o", "r", "space", "f", "o", "r", "space", 
                     "m", "e", "dot"]
        },
        4: {
            "sentence": "I just got home from work.",
            "keys": ["rshift", "i", "space", "j", "u", "s", "t", "space", "g", "o", "t", "space", 
                     "h", "o", "m", "e", "space", "f", "r", "o", "m", "space", "w", "o", "r", "k", "dot"]
        },
        5: {
            "sentence": "He is my best friend.",
            "keys": ["lshift", "h", "e", "space", "i", "s", "space", "m", "y", "space", 
                     "b", "e", "s", "t", "space", "f", "r", "i", "e", "n", "d", "dot"]
        },
        6: {
            "sentence": "Let's meet at nine o'clock.",
            "keys": ["lshift", "l", "e", "t", "apos", "s", "space", "m", "e", "e", "t", "space", 
                     "a", "t", "space", "n", "i", "n", "e", "space", "o", "apos", "c", "l", "o", "c", "k", "dot"]
        },
        7: {
            "sentence": "This is a rare opportunity.",
            "keys": ["rshift", "t", "h", "i", "s", "space", "i", "s", "space", "a", "space", 
                     "r", "a", "r", "e", "space", "o", "p", "p", "o", "r", "t", "u", "n", "i", "t", "y", "dot"]
        },
        8: {
            "sentence": "Did you see the news?",
            "keys": ["lshift", "d", "i", "d", "space", "y", "o", "u", "space", "s", "e", "e", "space", 
                     "t", "h", "e", "space", "n", "e", "w", "s", "lshift", "slash"]
        },
        9: {
            "sentence": "The case is closed now.",
            "keys": ["lshift", "t", "h", "e", "space", "c", "a", "s", "e", "space", "i", "s", "space", 
                     "c", "l", "o", "s", "e", "d", "space", "n", "o", "w", "dot"]
        },
        10: {
            "sentence": "Give me a ring later.",
            "keys": ["lshift", "g", "i", "v", "e", "space", "m", "e", "space", "a", "space", 
                     "r", "i", "n", "g", "space", "l", "a", "t", "e", "r", "dot"]
        },
        11: {
            "sentence": "Don't quit on this project.",
            "keys": ["lshift", "d", "o", "n", "apos", "t", "space", "q", "u", "i", "t", "space", 
                     "o", "n", "space", "t", "h", "i", "s", "space", "p", "r", "o", "j", "e", "c", "t", "dot"]
        },
        12: {
            "sentence": "I need to buy some milk.",
            "keys": ["lshift", "i", "space", "n", "e", "e", "d", "space", "t", "o", "space", 
                     "b", "u", "y", "space", "s", "o", "m", "e", "space", "m", "i", "l", "k", "dot"]
        },
        13: {
            "sentence": "The map is not loading.",
            "keys": ["lshift", "t", "h", "e", "space", "m", "a", "p", "space", "i", "s", "space", 
                     "n", "o", "t", "space", "l", "o", "a", "d", "i", "n", "g", "dot"]
        },
        14: {
            "sentence": "Is the file ready yet?",
            "keys": ["lshift", "i", "s", "space", "t", "h", "e", "space", "f", "i", "l", "e", "space", 
                     "r", "e", "a", "d", "y", "space", "y", "e", "t", "lshift", "slash"]
        },
        15: {
            "sentence": "We can win this game.",
            "keys": ["lshift", "w", "e", "space", "c", "a", "n", "space", "w", "i", "n", "space", 
                     "t", "h", "i", "s", "space", "g", "a", "m", "e", "dot"]
        }
    },

    # --- Part 2: Technical Commands (Syntax & Symbol Heavy) ---
    "Technical Commands": {
        16: {
            "sentence": "pip install numpy pandas",
            "keys": ["p", "i", "p", "space", "i", "n", "s", "t", "a", "l", "l", "space", 
                     "n", "u", "m", "p", "y", "space", "p", "a", "n", "d", "a", "s"]
        },
        17: {
            "sentence": "chmod 755 ./script.sh",
            "keys": ["c", "h", "m", "o", "d", "space", "7", "5", "5", "space", 
                     "dot", "slash", "s", "c", "r", "i", "p", "t", 
                     "dot", "s", "h"]
        },
        18: {
            "sentence": "select * from users where registrationYear > 2024;",
            "keys": ["s", "e", "l", "e", "c", "t", "space", "lshift", "8", "space", 
                     "f", "r", "o", "m", "space", "u", "s", "e", "r", "s", "space", 
                     "w", "h", "e", "r", "e", "space", 
                     "r", "e", "g", "i", "s", "t", "r", "a", "t", "i", "o", "n", "lshift", "y", "e", "a", "r",
                    "space", "rshift", "dot", "space", 
                     "2", "0", "2", "4", "semicolon"]
        },
        19: {
            "sentence": 'print(f"Accuracy: {acc:.2f}%")',
            "keys": ["p", "r", "i", "n", "t", "lshift", "9", "f", "rshift", "apos", 
                     "rshift", "a", "c", "c", "u", "r", "a", "c", "y", "lshift", "semicolon", "space", 
                     "lshift", "bracketopen", "a", "c", "c", "lshift", "semicolon", "dot", "2", "f", "lshift", "bracketclose", 
                     "lshift", "5", "lshift", "apos", "lshift", "0"]
        },
        20: {
            "sentence": "git status",
            "keys": ["g", "i", "t", "space", "s", "t", "a", "t", "u", "s"]
        },
        21: {
            "sentence": "sudo apt update",
            "keys": ["s", "u", "d", "o", "space", "a", "p", "t", "space", 
                     "u", "p", "d", "a", "t", "e"]
        },
        22: {
            "sentence": "ssh root@192.168.1.55",
            "keys": ["s", "s", "h", "space", "r", "o", "o", "t", "lshift", "2", 
                     "1", "9", "2", "dot", "1", "6", "8", "dot", "1", "dot", "5", "5"]
        },
        23: {
            "sentence": "docker pull nginx",
            "keys": ["d", "o", "c", "k", "e", "r", "space", "p", "u", "l", "l", "space", 
                     "n", "g", "i", "n", "x"]
        },
        24: {
            "sentence": 'const data = { id: 1, name: "admin" };',
            "keys": ["c", "o", "n", "s", "t", "space", "d", "a", "t", "a", "space", "equal", "space", 
                     "lshift", "bracketopen", "space", 
                     "i", "d", "lshift", "semicolon", "space", "1", "comma", "space", 
                     "n", "a", "m", "e", "lshift", "semicolon", "space", 
                     "rshift", "apos", "a", "d", "m", "i", "n", "lshift", "apos", "space", 
                     "rshift", "bracketclose", "semicolon"]
        },
        25: {
            "sentence": "https://www.google.com/search?q=test",
            "keys": ["h", "t", "t", "p", "s", "lshift", "semicolon", "slash", "slash", 
                     "w", "w", "w", "dot", "g", "o", "o", "g", "l", "e", "dot", "c", "o", "m", "slash", 
                     "s", "e", "a", "r", "c", "h", "lshift", "slash", "q", "equal", "t", "e", "s", "t"]
        },
        26: {
            "sentence": "for i in range(10): print(i * 2)",
            "keys": ["f", "o", "r", "space", "i", "space", "i", "n", "space", 
                     "r", "a", "n", "g", "e", "lshift", "9", "1", "0", "rshift", "0", "lshift", "semicolon", "space", 
                     "p", "r", "i", "n", "t", "lshift", "9", "i", "space", "rshift", "8", "space", "2", "lshift", "0"]
        },
        27: {
            "sentence": "cat /var/log/syslog",
            "keys": ["c", "a", "t", "space", 
                     "slash", "v", "a", "r", "slash", "l", "o", "g", "slash", "s", "y", "s", "l", "o", "g"]
        },
        28: {
            "sentence": "nano /etc/hosts",
            "keys": ["n", "a", "n", "o", "space", 
                     "slash", "e", "t", "c", "slash", "h", "o", "s", "t", "s"]
        },
        29: {
            "sentence": 'ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"',
            "keys": ["i", "p", "v", "6", "space", "equal", "space", 
                     "lshift", "apos", 
                     "2", "0", "0", "1", "lshift", "semicolon", "0", "d", "b", "8", "rshift", "semicolon", 
                     "8", "5", "a", "3", "lshift", "semicolon", 
                     "0", "0", "0", "0", "lshift", "semicolon", "0", "0", "0", "0", "rshift", "semicolon", 
                     "8", "a", "2", "e", "lshift", "semicolon", 
                     "0", "3", "7", "0", "lshift", "semicolon", 
                     "7", "3", "3", "4", "lshift", "apos"]
        },
        30: {
            "sentence": "mkdir projects",
            "keys": ["m", "k", "d", "i", "r", "space", "p", "r", "o", "j", "e", "c", "t", "s"]
        }
    }
}


SPECIAL_KEY_MAP = {
        "bracketopen":  ("[", "{"),
        "bracketclose": ("]", "}"),
        "semicolon":    (";", ":"),
        "quote":        ("'", '"'),
        "apos":         ("'", '"'),
        "comma":        (",", "<"),
        "dot":          (".", ">"),
        "period":       (".", ">"),
        "slash":        ("/", "?"),
        "backslash":    ("\\", "|"),
        "equal":        ("=", "+"),
        "minus":        ("-", "_"),
        "dash":         ("-", "_"),
        "space":        (" ", " "),
        "tab":          ("\t", "\t"),
        
        "1": ("1", "!"), "2": ("2", "@"), "3": ("3", "#"),
        "4": ("4", "$"), "5": ("5", "%"), "6": ("6", "^"),
        "7": ("7", "&"), "8": ("8", "*"), "9": ("9", "("),
        "0": ("0", ")")
    }

KEY_NORMALIZATION = {
    "period": "dot", 
    "return": "enter",
    "spacebar": "space",
}