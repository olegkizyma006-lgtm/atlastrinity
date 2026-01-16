import asyncio
import re

async def test_sentence_splitting():
    print("=== Testing TTS Sentence Splitting Logic ===")
    
    test_texts = [
        "Привіт! Як справи? Я Атлас.",
        "Це довге речення, яке не повинно розриватися. А це вже друге речення.",
        "Що це? Де я! Ого...",
        "Текст без крапок в кінці",
        "Текст з скороченнями і т.д. Але крапка в кінці є."
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        # Replication of splitting logic from tts.py
        chunks = re.split(r'([.!?]+(?:\s+|$))', text)
        processed_chunks = []
        for i in range(0, len(chunks)-1, 2):
            processed_chunks.append(chunks[i] + chunks[i+1])
        if len(chunks) % 2 == 1 and chunks[-1]:
             processed_chunks.append(chunks[-1])
        
        initial_chunks = [c.strip() for c in processed_chunks if c.strip()]
        
        # Merging logic from tts.py
        min_len = 40
        refined_chunks = []
        temp_chunk = ""
        for chunk in initial_chunks:
            if temp_chunk:
                temp_chunk += " " + chunk
            else:
                temp_chunk = chunk
            if len(temp_chunk) >= min_len:
                refined_chunks.append(temp_chunk)
                temp_chunk = ""
        if temp_chunk:
            if refined_chunks:
                refined_chunks[-1] += " " + temp_chunk
            else:
                refined_chunks.append(temp_chunk)
        final_chunks = refined_chunks
        
        print(f"Refined into {len(final_chunks)} chunks:")
        for idx, chunk in enumerate(final_chunks):
            print(f"  {idx+1}: {chunk} (Length: {len(chunk)})")

async def simulate_playback():
    print("\n=== Simulating Queued Playback Loop ===")
    chunks = ["Перше речення.", "Друге речення!", "Третє речення?"]
    
    for idx, chunk in enumerate(chunks):
        print(f"Generating chunk {idx+1}/{len(chunks)}: {chunk}")
        await asyncio.sleep(0.2) # Simulate generation
        print(f"Speaking chunk {idx+1}/{len(chunks)}: {chunk}")
        await asyncio.sleep(0.5) # Simulate playback
        print(f"Finished chunk {idx+1}")

if __name__ == "__main__":
    asyncio.run(test_sentence_splitting())
    asyncio.run(simulate_playback())
