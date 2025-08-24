MAP = {
    "lctrl": "ctrl",
    "lcmd": "cmd",
    "lalt": "alt",
    "lshift": "shift",
    "ralt": "alt",
    "rctrl": "ctrl",
    "rshift": "shift",
    "rcmd": "cmd",
    "bracketclose": "bracket",
    "bracketopen": "bracket",
}

TEST_DATASETS = [
    "practical",
    "noiseless",
    "mka",
    "custom_mac",
    "custom_dishwasher",
    "custom_open_window",
    "custom_washing_machine",
]

DATASET_GROUPS = {
    "all_w_custom": ["mka", "practical", "noiseless", "custom_mac"],
    "all_w_custom_noisy": [
        "mka",
        "practical",
        "noiseless",
        "custom_mac",
        "custom_dishwasher",
        "custom_open_window",
        "custom_washing_machine",
    ],
    "all": ["mka", "practical", "noiseless"],
    "custom_noisy": [
        "custom_dishwasher",
        "custom_open_window",
        "custom_washing_machine",
    ],
    "custom": [
        "custom_mac",
        "custom_dishwasher",
        "custom_open_window",
        "custom_washing_machine",
    ],
}
EXCLUDED_KEYS = set(["fn", "start"])
ALPHANUMERIC_KEYS = set("abcdefghijklmnopqrstuvwxyz0123456789")
