from torch.utils.data import Dataset

import utils.utils as utils


class GeneralDataset(Dataset):
    def __init__(self, main_dataset, user_profile_map, asin_map, prompt_generator):
        self.main_dataset = main_dataset
        self.user_profile_map = user_profile_map
        self.asin_map = asin_map
        self.prompt_generator = prompt_generator

    def __getitem__(self, index):
        sample = self.main_dataset[index]
        user_id = sample["user_id"]
        profile = self.user_profile_map[user_id]
        for p in profile:
            asin = p["asin"]
            item_title, description = self.asin_map[asin]
            p["item_title"] = item_title
            p["description"] = description
        data = sample["data"]
        asin = data["asin"]
        item_title, description = self.asin_map[asin]
        text, rating, review_title = data["text"], data["rating"], data["title"]
        inp_creator, summ_creator, diff_inp = self.prompt_generator(user_id, item_title, description,
                                                                    rating, review_title, profile)
        return inp_creator, summ_creator, diff_inp, text

    def __len__(self):
        return len(self.main_dataset)


def create_prompt_generator(num_retrieved,
                            user_profile_map,
                            asin_reviewers_map,
                            embedder):

    def prompt(user_id, item_title, description, rating, review_title, profile):
        return create_data(num_retrieved, user_id, item_title, 
                           description, rating, review_title,
                           profile, user_profile_map, 
                           asin_reviewers_map, embedder)
    return prompt


def create_data(num_retrieved, user_id, item_title, description, rating, review_title,
                profile, user_profile_map, asin_reviewers_map, embedder):

    diff_profile = [
        prof for prof in profile
        if len(sorted(asin_reviewers_map[prof["asin"]] - {user_id})) >= 4
    ]
    selected_profile = utils.get_selected_profile(profile, review_title,
                                                   description, num_retrieved)
    selected_diff_profile = utils.get_selected_profile(diff_profile, review_title, 
                                                       description, num_retrieved)

    def create_diff_inp(item_title, description, cur, others):
        cur_rating, cur_title, cur_text = cur
        item_context = (
            f"[Item Information]:\n"
            f"- [Item Title]: {item_title}\n"
            f"- [Item Description]: {description}\n"
        )
        current_reviewer = (
            f"\n[Review by the current user]:\n"
            f"- [User's Rating]: {cur_rating}\n"
            f"- [Review Title]: {cur_title}\n"
            f"- [Review Text]: {cur_text}\n"
        )
        other_reviewers = []
        for idx, other in enumerate(others):
            other_rating, other_title, other_text = other
            other_reviewer = (
                f"\n[Review by the other user {idx + 1}]:\n"
                f"- [User's Rating]: {other_rating}\n"
                f"- [Review Title]: {other_title}\n"
                f"- [Review Text]: {other_text}\n"
            )
            other_reviewers.append(other_reviewer)
        prompt = item_context + current_reviewer + "".join(other_reviewers)
        return prompt

    def create_summ_inp(diff_outputs):
        prompts = []
        for prof, diff_output in zip(selected_diff_profile, diff_outputs):
            p_item_title = prof["item_title"]
            p_desc = prof["description"]
            prompt = (
                f"[Item Title]: {p_item_title}\n"
                f"[Item Description]: {p_desc}\n"
                f"[Differences between the current user's review and the other user's reviews]:"
                f"\n{diff_output}\n"
            )
            prompts.append(prompt)
        past_reviews = "".join([
            f"[Review {idx + 1}]:\n"
            f"- [Item Title]: {p['item_title']}\n"
            f"- [Item Description]: {p['description']}\n"
            f"- [Review Rating]: {p['rating']}\n"
            f"- [Review Title]: {p['title']}\n"
            f"- [Review Text]: {p['text']}\n"
            for idx, p in enumerate(selected_profile)
        ])
        return "\n".join(prompts) + f"\n[User's Past Reviews]:\n{past_reviews}\n"

    def create_inp(summary):
        past_reviews = "".join([
            f"[Review {idx + 1}]:\n"
            f"- [Item Title]: {p['item_title']}\n"
            f"- [Item Description]: {p['description']}\n"
            f"- [Review Rating]: {p['rating']}\n"
            f"- [Review Title]: {p['title']}\n"
            f"- [Review Text]: {p['text']}\n"
            for idx, p in enumerate(selected_profile)
        ])
        prefix = "[Summary]:"
        if summary.startswith(prefix):
            summary = summary[len(prefix):].strip()
        prompt = (
            f"[Item Title]: {item_title}\n"
            f"[Item Description]: {description}\n"
            f"[User's Past Reviews]:\n{past_reviews}\n"
            f"[User's Profile Summary]:\n{summary}\n"
            f"[Output Review Rating]: {rating}\n"
            f"[Output Review Title]: {review_title}\n"
        )
        return prompt

    diff_inputs = []
    for prof in selected_diff_profile:
        p_asin = prof["asin"]
        p_item_title = prof["item_title"]
        p_desc = prof["description"]
        p_rating = prof["rating"]
        p_review_title = prof["title"]
        p_text = prof["text"]
        reviewers = sorted(asin_reviewers_map[p_asin])
        reviewers.remove(user_id)
        cur = (p_rating, p_review_title, p_text)
        others = []
        for reviewer in reviewers:
            other_reviewer_profile = user_profile_map[reviewer]
            for pp in other_reviewer_profile:
                if pp["asin"] == p_asin:
                    others.append((pp["rating"], pp["title"], pp["text"]))
                    break
        others = utils.get_others_by_kmeans(p_text, others, embedder, n=4)
        diff_inp = create_diff_inp(p_item_title, p_desc, cur, others)
        diff_inputs.append(diff_inp)
    return create_inp, create_summ_inp, diff_inputs
