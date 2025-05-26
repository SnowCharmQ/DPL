import os
import re
import csv
import torch
import random
import numpy as np
from filelock import FileLock

from rank_bm25 import BM25Okapi
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


def get_others_by_random(cur_text, others, embedder, n=4):
    return random.sample(others, n)


def get_others_by_sim_ranking(cur_text, others, embedder, n=4):
    if isinstance(embedder, SentenceTransformer):
        others_text = [x[2] for x in others]
        cur_text_embedding = embedder.encode(cur_text,
                                            batch_size=32)
        others_text_embedding = embedder.encode(others_text,
                                                batch_size=32)
        similarities = embedder.similarity(cur_text_embedding,
                                        others_text_embedding)
        n_indices = torch.topk(similarities, n, largest=False).indices.tolist()[0]
        return [others[i] for i in n_indices]


def get_others_by_kmeans(cur_text, others, embedder, n=4, method='closest'):
    results = []
    whole_text = [cur_text] + [x[2] for x in others]
    whole_text_embedding = embedder.encode(whole_text, batch_size=32)
    
    kmeans = KMeans(n_clusters=n+1, random_state=42).fit(whole_text_embedding)
    labels = kmeans.labels_
    target_label = labels[0]
    cluster_centers = kmeans.cluster_centers_
    
    cluster_dict = {i: [] for i in range(n+1)}
    for i, label in enumerate(labels[1:], start=1):
        cluster_dict[label].append((i-1, whole_text_embedding[i]))
    
    available_clusters = [c for c in cluster_dict if c != target_label and cluster_dict[c]]
    for cluster in available_clusters:
        cluster_vectors = cluster_dict[cluster]
        if method == 'closest':
            distances = [np.linalg.norm(vec[1] - cluster_centers[cluster]) for vec in cluster_vectors]
            closest_idx = cluster_vectors[np.argmin(distances)][0]
            results.append(others[closest_idx])
        else:
            random_idx = random.choice(cluster_vectors)[0]
            results.append(others[random_idx])
    
    return results


def get_selected_profile(profile, review_title, description, num_retrieved):
    corpus = [f'{x["title"]} {x["description"]}' for x in profile]
    query = f'{review_title} {description}'
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    return bm25.get_top_n(tokenized_query, profile, n=num_retrieved)


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def postprocess_output(output):
    prefix = "[Review]:"
    if output.startswith(prefix):
        return output[len(prefix):].strip()
    return output


def write_to_csv(method, metric, value):
    file_path = "../result.csv"
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(file_path):
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file))
        else:
            reader = []
        if not reader:
            reader.append(["method"])
        headers = reader[0]
        methods = [row[0] for row in reader[1:]]
        if metric not in headers:
            headers.append(metric)
            for row in reader[1:]:
                row.append("")
        if method not in methods:
            reader.append([method] + ["" for _ in range(len(headers) - 1)])
        for row in reader:
            while len(row) < len(headers):
                row.append("")
        method_index = methods.index(
            method) + 1 if method in methods else len(reader) - 1
        metric_index = headers.index(metric)
        reader[method_index][metric_index] = value
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
