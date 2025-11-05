import pandas as pd
from datetime import timedelta
import glob
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor

def generate_examples_for_author(df, author_name):
    """Generate training examples for a specific author from a dataframe."""
    examples = []
    last_used_idx = -1  # Track the last index we used to avoid overlap

    # Find all messages from this author
    author_indices = df[df['author_name'] == author_name].index.tolist()

    for idx in author_indices:
        # Skip if we've already used this message in a previous example
        if idx <= last_used_idx:
            continue

        # Only create example if previous message is NOT from this author
        # (i.e., this is the START of an author response burst)
        if idx > 0 and df.loc[idx - 1, 'author_name'] == author_name:
            continue

        # Get the timestamp of this author's message
        current_time = df.loc[idx, 'timestamp']

        # Get 5 minutes of context BEFORE this message
        context_start = current_time - timedelta(minutes=5)
        context_mask = (df['timestamp'] >= context_start) & (df['timestamp'] < current_time)

        # Exclude any messages that were used in previous examples
        if last_used_idx >= 0:
            context_mask = context_mask & (df.index > last_used_idx)

        context_messages = df[context_mask]

        # Skip if no context
        if len(context_messages) == 0:
            continue

        # Build context: concatenate messages with newlines
        context_lines = []
        for _, row in context_messages.iterrows():
            context_lines.append(f"[{row['author_name']}]: {row['content']}")
        context = "\n".join(context_lines)

        # Build response: get up to 5 sequential messages from this author starting from current
        response_lines = []
        for i in range(5):
            if idx + i < len(df) and df.loc[idx + i, 'author_name'] == author_name:
                response_lines.append(df.loc[idx + i, 'content'])
                last_used_idx = idx + i  # Mark this index as used
            else:
                break

        # Skip if no response
        if len(response_lines) == 0:
            continue

        response = "\n".join(response_lines)

        examples.append({
            'prompt': context,
            'completion': response
        })

    return examples


def process_file_for_author(parquet_file, author):
    """Process a single file for a single author."""
    file_name = os.path.basename(parquet_file)

    # Read and prepare dataframe
    df = pd.read_parquet(parquet_file)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Generate examples
    examples = generate_examples_for_author(df, author)

    # Add metadata
    for ex in examples:
        ex['author'] = author
        ex['source_file'] = file_name

    return file_name, author, examples


async def main():
    """Main async processing loop."""
    discord_dir = "/Users/marcoseoane/Desktop/sft/discord"
    output_dir = "/Users/marcoseoane/Desktop/sft/training_data"
    os.makedirs(output_dir, exist_ok=True)

    # Get all parquet files
    parquet_files = glob.glob(os.path.join(discord_dir, "*.parquet"))

    # Authors to process
    authors = ['edgefills', 'kgb2938']

    print(f"Processing {len(parquet_files)} parquet files for authors: {', '.join(authors)}")
    print("=" * 80)

    # Create tasks for all file-author combinations
    tasks = []
    for parquet_file in sorted(parquet_files):
        for author in authors:
            tasks.append((parquet_file, author))

    # Process all tasks in parallel using ProcessPoolExecutor
    all_examples = []
    loop = asyncio.get_event_loop()

    with ProcessPoolExecutor() as executor:
        futures = [
            loop.run_in_executor(executor, process_file_for_author, pf, auth)
            for pf, auth in tasks
        ]

        for future in asyncio.as_completed(futures):
            file_name, author, examples = await future
            if examples:
                all_examples.extend(examples)
                print(f"✓ {file_name} - {author}: {len(examples)} examples")

    print("\n" + "=" * 80)
    print(f"Total examples generated: {len(all_examples)}")

    # Create DataFrame and save
    if all_examples:
        examples_df = pd.DataFrame(all_examples)
        output_file = os.path.join(output_dir, "training_examples.parquet")
        examples_df.to_parquet(output_file, index=False)

        print(f"Saved to: {output_file}")
        print(f"DataFrame shape: {examples_df.shape}")
        print(f"Columns: {list(examples_df.columns)}")

        # Show distribution by author
        print("\n" + "=" * 80)
        print("Distribution by author:")
        print(examples_df['author'].value_counts())

        print("\n" + "=" * 80)
        print("First example:")
        print("=" * 80)
        print(f"\nAUTHOR: {examples_df.iloc[0]['author']}")
        print(f"SOURCE: {examples_df.iloc[0]['source_file']}")
        print(f"\nPROMPT:\n{examples_df.iloc[0]['prompt'][:500]}...")
        print(f"\nCOMPLETION:\n{examples_df.iloc[0]['completion'][:500]}...")
    else:
        print("No examples generated!")


if __name__ == "__main__":
    asyncio.run(main())
