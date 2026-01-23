import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.brain.memory import long_term_memory


async def cleanup(total_wipe=False):
    print("🚀 Starting Memory Cleanup...")

    # 1. Delete specific hallucinations
    hallucinations = [
        "Сподівайся, як обходить",
        "я не маю прямого доступу до актуальних метеорологічних даних",
        "я не можу надати точний прогноз погоди",
        "нажаль я не маю доступу",
        "не маю доступу до інтернет",
        "я не маю прямого доступу до інтернету",
    ]

    for h in hallucinations:
        print(f"🔍 Searching for: {h}...")
        deleted = await long_term_memory.delete_specific_memory("conversations", h)
        if deleted:
            print(f"✅ Removed from conversations: ({deleted} entries)")

        deleted_lessons = await long_term_memory.delete_specific_memory("lessons", h)
        if deleted_lessons:
            print(f"✅ Removed from lessons: ({deleted_lessons} entries)")

    # 2. Clear all learning (if flag is set)
    if total_wipe:
        print("\n⚠️ PERFORMING TOTAL VECTOR MEMORY WIPE...")
        success = await long_term_memory.clear_all_memory()
        if success:
            print("✨ ALL VECTOR MEMORY CLEARED SUCCESSFULLY.")
        else:
            print("❌ Failed to clear memory.")
    else:
        print("\n💡 Tip: Run with --total to wipe all long-term memory.")

    print("\n✅ Cleanup finished.")


if __name__ == "__main__":
    import sys

    total = "--total" in sys.argv
    asyncio.run(cleanup(total_wipe=total))
