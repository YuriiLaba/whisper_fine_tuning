import os

train_id = "1NE64maSRrg1M2upJLkrSDy_vKQ674yIc"
eval_id = "1dK2Mij3sihwAq5ob_ST2tYa5VkGyDndZ?"

train_name = "dataset"
eval_name = "eval_dataset"

os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={eval_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={eval_id}" -O {eval_name} && rm -rf /tmp/cookies.txt""")
os.system(f"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={train_id}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={train_id}" -O {train_name} && rm -rf /tmp/cookies.txt""")