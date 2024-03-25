# cattrack
cat track is a small script to scrape, classify, and then notify prospective cat
owners of newly added cats. My local humane society posts listings that contain
a hidden div with some model that I can translate into mine. I scrape the page
in one go, then save each cats picture in the Cat Cache (catche), then analyze
the picture with a model trained on the
[Cat Breeds Dataset](https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset).

## Dataset
The dataset has quite a lot of notable issues, chiefly being that the data is
sourced from advertisers trying to make their cats attractive. It is also
unbalanced with less exotic breeds. Some breeds don't have enough data to train
with. There's even a dog in the set! However, I'm approaching this project with
the mentality of "good enough". :D

## Quickstart
### Configure
Edit the contents of `config.yaml.example` and save as `config.yaml`.
### Train the AI
Run src/train.py to generate a model. This can take several hours.
### Run the scraper
Run main.py. Maybe put it in a bash loop to periodically check every hour or so.
