python ../llava/serve/cli.py \
--model-path ../llava-vistral-merged \
--image-file https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg \
--conv-mode vistral-it \
--load-8bit
