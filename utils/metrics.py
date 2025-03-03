def build_llm_evaluate_prompt(pd, gt, data):
    item_title, item_description, review_rating, review_title = data
    return (
        f'[Item Title]: {item_title}\n'
        f'[Item Description]: {item_description}\n'
        f'[Review Rating]: {review_rating}\n'
        f'[Review Title]: {review_title}\n'
        f"[User's real review]: {gt}\n"
        f"[AI-generated review]: {pd}\n"
        f"[Score]: "
    )
