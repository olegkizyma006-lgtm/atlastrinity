#!/usr/bin/env python3
import sys, asyncio, json
sys.path.insert(0, 'src')
from brain.mcp_manager import mcp_manager
from brain.agents.tetyana import Tetyana

async def run():
    print('--- Creating note for step_424')
    note = await mcp_manager.call_tool('notes','create_note', {
        'title':'Test Grisha Note for Tetyana',
        'content':'This is a test rejection note from Grisha for step 424.',
        'category':'verification_report',
        'tags':['grisha','step_424']
    })
    print('create_note ->', repr(note)[:400])

    print('\n--- Searching notes by tag step_424')
    search = await mcp_manager.call_tool('notes','search_notes', {'tags':['step_424'], 'limit':5})
    print('search_notes ->', getattr(search, 'structuredContent', None) or repr(search)[:400])

    print('\n--- Tetyana reading Grisha feedback for step 424')
    tet = Tetyana(model_name='grok-code-fast-1')
    feedback = await tet.get_grisha_feedback(424)
    print('Tetyana.get_grisha_feedback ->', feedback[:800] if feedback else feedback)

    print('\n--- Creating memory entity tetyana_test_entity')
    memc = await mcp_manager.call_tool('memory','create_entities', {'entities':[{'name':'tetyana_test_entity','entityType':'test','observations':['created by tetyana for grisha']}]})
    print('memory.create_entities ->', getattr(memc, 'content', None) or repr(memc)[:400])

    print('\n--- Searching memory for tetyana_test_entity')
    mems = await mcp_manager.call_tool('memory','search_nodes', {'query':'tetyana_test_entity','limit':5})
    print('memory.search_nodes ->', getattr(mems, 'content', None) or repr(mems)[:400])

    print('\n--- Getting entity tetyana_test_entity')
    ent = await mcp_manager.call_tool('memory','get_entity', {'name':'tetyana_test_entity'})
    print('memory.get_entity ->', getattr(ent, 'content', None) or repr(ent)[:400])

    await mcp_manager.cleanup()

if __name__ == '__main__':
    asyncio.run(run())
