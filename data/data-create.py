import re
import argparse

from collections import defaultdict, Counter
from datasets import load_dataset, Dataset, DatasetDict

parser = argparse.ArgumentParser()
parser.add_argument("--category", type=str)
parser.add_argument("--push_to_hub", action="store_true")
args = parser.parse_args()

category = args.category

review_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                              f"raw_review_{category}",
                              split="full",
                              trust_remote_code=True)
meta_dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023",
                            f"raw_meta_{category}",
                            split="full",
                            trust_remote_code=True)


def review_filter(x):
    return (
        x['text'] and len(x['text']) >= 200 and
        x['rating'] and x['title'] and x['timestamp']
    )


def meta_filter(x):
    return (
        2000 >= len(create_description(x)) >= 100 and
        x['title'] and x['categories'] and x['rating_number']
    )


def create_description(x):
    if not x['description'] or (x['description'][0] == 'Product Description' and len(x['description']) == 1):
        return ''
    elif x['description'][0] == 'Product Description' and len(x['description']) > 1:
        return ' '.join(x['description'][1:])
    else:
        return ' '.join(x['description'])


meta_columns = meta_dataset.column_names
meta_columns.remove("title")
meta_columns.remove("description")
meta_dataset = meta_dataset.filter(lambda x: meta_filter(x),
                                   num_proc=64,
                                   load_from_cache_file=False).map(lambda x: {'asin': x['parent_asin'],
                                                                              'title': x['title'],
                                                                              'description': create_description(x)},
                                                                   num_proc=64,
                                                                   remove_columns=meta_columns,
                                                                   load_from_cache_file=False)

asins = meta_dataset['asin']
review_columns = review_dataset.column_names
review_columns.remove("user_id")
review_columns.remove("asin")
review_columns.remove("title")
review_columns.remove("text")
review_columns.remove("rating")
review_columns.remove("timestamp")
review_dataset = review_dataset.filter(lambda x: review_filter(x),
                                       num_proc=64,
                                       load_from_cache_file=False).map(lambda x: {'user_id': x['user_id'],
                                                                                  'asin': x['parent_asin'],
                                                                                  'title': x['title'],
                                                                                  'text': x['text'],
                                                                                  'rating': x['rating'],
                                                                                  'timestamp': x['timestamp']},
                                                                       num_proc=64,
                                                                       remove_columns=review_columns,
                                                                       load_from_cache_file=False)
review_dataset = review_dataset.filter(lambda x: x['asin'] in asins,
                                       num_proc=64, load_from_cache_file=False)

cnt = 0
while True:
    cnt += 1
    user_review = defaultdict(list)
    asin_user = defaultdict(set)

    def recalculate(example):
        user_id = example['user_id']
        asin = example['asin']
        user_review[user_id].append({
            'rating': example['rating'],
            'title': example['title'],
            'text': example['text'],
            'asin': example['asin'],
            'timestamp': example['timestamp']
        })
        asin_user[asin].add(user_id)
        return example

    new_review_dataset = review_dataset.map(recalculate,
                                            load_from_cache_file=False)

    for user_id, reviews in user_review.items():
        sorted_reviews = sorted(reviews,
                                key=lambda x: x['timestamp'],
                                reverse=True)
        seen_asin = set()
        seen_text = set()
        unique_reviews = []
        for review in sorted_reviews:
            if review['asin'] not in seen_asin and review['text'] not in seen_text:
                seen_asin.add(review['asin'])
                seen_text.add(review['text'])
                unique_reviews.append(review)
        user_review[user_id] = unique_reviews

    asin_user_count = Counter({asin: len(users)
                              for asin, users in asin_user.items()})
    user_review_count = Counter({user: len(reviews)
                                for user, reviews in user_review.items()})

    for asin in list(asin_user.keys()):
        if asin_user_count[asin] < 8:
            del asin_user[asin]

    for user in list(user_review.keys()):
        if user_review_count[user] < 18 or user_review_count[user] > 500:
            del user_review[user]

    new_review_dataset = new_review_dataset.filter(
        lambda x: x['asin'] in asin_user and x['user_id'] in user_review,
        load_from_cache_file=False)

    print(len(new_review_dataset), len(review_dataset))
    if len(new_review_dataset) == len(review_dataset):
        print(f"Finished in {cnt} iterations")
        break
    review_dataset = new_review_dataset

meta_dataset = meta_dataset.filter(lambda x: x['asin'] in asin_user)

user_review = defaultdict(list)
asin_user = defaultdict(set)


def postprocess(example):
    user_id = example['user_id']
    asin = example['asin']
    user_review[user_id].append({
        'rating': example['rating'],
        'title': example['title'],
        'text': re.sub(r'\s+', ' ', example['text'].strip()),
        'asin': example['asin'],
        'timestamp': example['timestamp']
    })
    asin_user[asin].add(user_id)
    return example


review_dataset = review_dataset.map(postprocess,
                                    load_from_cache_file=False)
for user_id, reviews in user_review.items():
    sorted_reviews = sorted(
        reviews, key=lambda x: x['timestamp'], reverse=True)
    seen_asin = set()
    seen_text = set()
    unique_reviews = []
    for review in sorted_reviews:
        if review['asin'] not in seen_asin and review['text'] not in seen_text:
            seen_asin.add(review['asin'])
            seen_text.add(review['text'])
            unique_reviews.append(review)
    user_review[user_id] = unique_reviews

asin_user_new = defaultdict(set)
user_review_split = defaultdict(tuple)
for user_id, reviews in user_review.items():
    reviews = sorted(reviews, key=lambda x: x['timestamp'])
    profile_reviews = reviews[:-10]
    input_reviews = reviews[-10:]
    user_review_split[user_id] = (profile_reviews, input_reviews)
    for review in profile_reviews:
        asin_user_new[review['asin']].add(user_id)

cnt = 0
while True:
    cnt += 1
    asin_user_count = Counter({asin: len(users)
                              for asin, users in asin_user_new.items()})
    for asin in list(asin_user_new.keys()):
        if asin_user_count[asin] < 5:
            del asin_user_new[asin]
    user_review_split_new = {}
    for user_id, (profile_reviews, input_reviews) in user_review_split.items():
        new_reviews = []
        for review in profile_reviews:
            if review['asin'] in asin_user_new:
                new_reviews.append(review)
        if len(new_reviews) < 8:
            continue
        user_review_split_new[user_id] = (new_reviews, input_reviews)
    asin_user_new = defaultdict(set)
    for user_id, (new_reviews, _) in user_review_split_new.items():
        for review in new_reviews:
            asin_user_new[review['asin']].add(user_id)
    print(len(user_review_split), len(user_review_split_new))
    if len(user_review_split) == len(user_review_split_new):
        print(f"Finished in {cnt} iterations")
        break
    user_review_split = user_review_split_new

asin_reviewers_map = defaultdict(set)

train_dataset = []
val_dataset = []
test_dataset = []
for user_id, (profile_reviews, input_reviews) in user_review_split_new.items():
    for i in range(len(input_reviews) - 2):
        train_data = input_reviews[i]
        new_reviews = profile_reviews + input_reviews[:i]
        train_dataset.append((user_id, new_reviews, train_data))
    val_dataset.append((user_id, profile_reviews + input_reviews[:-2],
                        input_reviews[-2]))
    test_dataset.append((user_id, profile_reviews + input_reviews[:-1],
                        input_reviews[-1]))

user_ids, profiles, train_data = zip(*[
    (user_id, reviews, train_review)
    for user_id, reviews, train_review in train_dataset
])
user_ids = list(user_ids)
profiles = list(profiles)
train_data = list(train_data)
data_dict = {
    "user_id": user_ids,
    "profile": profiles,
    "data": train_data
}
train_dataset = Dataset.from_dict(data_dict)

user_ids, profiles, val_data = zip(*[
    (user_id, reviews, val_review)
    for user_id, reviews, val_review in val_dataset
])
user_ids = list(user_ids)
profiles = list(profiles)
val_data = list(val_data)
data_dict = {
    "user_id": user_ids,
    "profile": profiles,
    "data": val_data
}
val_dataset = Dataset.from_dict(data_dict)

user_ids, profiles, test_data = zip(*[
    (user_id, reviews, test_review)
    for user_id, reviews, test_review in test_dataset
])
user_ids = list(user_ids)
profiles = list(profiles)
test_data = list(test_data)
data_dict = {
    "user_id": user_ids,
    "profile": profiles,
    "data": test_data
}
test_dataset = Dataset.from_dict(data_dict)

main_dataset = DatasetDict({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset
})
meta_dataset = DatasetDict(
    {
        "full": meta_dataset
    }
)

main_dataset.save_to_disk(f"../DPL-main/{category}")
meta_dataset.save_to_disk(f"../DPL-meta/{category}")
