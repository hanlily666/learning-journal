### Purpose

Pick what is core to the session's goal so I can review it later. The goal is usually my
original question, summarized. Pick:

- **gap**: something I didn't know: the answer to my question, and anything the assistant
  pointed out that I hadn't thought of.
- **concept**: the concept that is required to fully understand the original questions; the concept is still required for AI learning although might not be a hard requirement to this session's understanding.
- **explore**: a topic/discussion that is expanded on the original question which is hardly required to understand as a concept to help build understanding and fully grasp the knowledge of the original question.
- **source**: a direct link (URL) to an external source that the assistant gave. A source
  mentioned without a link, such as "the MCP spec", is not a source item.

Copy each item's text from the chat as it is written: the passage that explains it, word for
word, not a summary. If something is only named, such as a concept mentioned in passing, copy
the name. Give the reason in a few words.

If the same idea is explained more than once, pick it once and copy the passage that explains
it most fully.

Don't pick detailed content: code, commands, single parameters, or long instructions on how
to learn something. Copy the sentences that explain the idea the code shows, not the code. A session can have few items,
or none.

### Examples

Each example shows the messages as you see them, then what to submit. The "not picked" part
only explains what the example leaves out and why. Never submit it; the user never sees it.

#### Example 1: a question with an answer I didn't expect

<message id="101" role="user">
what does sampling in MCP mean?
</message>
<message id="102" role="assistant">
Mid-execution, the server decides it needs an LLM completion to do its job (e.g., summarize a
doc it just fetched, decide which of several actions to take, draft some text).

One thing worth knowing since you're building agent systems: as of the 2026-07-28 MCP spec
revision (SEP-2577), sampling was formally deprecated — along with Roots and Logging — due to
low client adoption and the complexity of implementing it correctly (human-in-the-loop, model
selection, security). It's still fully functional for at least a year from that revision, but
the guidance going forward is for servers to integrate directly with an LLM provider API
rather than relying on sampling.

**Why generation is called "sampling" in ML**

When an LLM produces text, at each step it outputs a probability distribution over possible
next tokens, and the actual token is chosen by sampling from that distribution (that's what
temperature, top-p, and top-k control). So "sampling" is standard shorthand for "run inference
and get a generated output."
</message>

subject: MCP sampling
summary: I asked what sampling means in MCP. It lets a server ask the client's model for a
completion mid-task, but it was deprecated in the 2026-07-28 spec revision.
items:

- gap: Mid-execution, the server decides it needs an LLM completion to do its job (e.g.,
  summarize a doc it just fetched, decide which of several actions to take, draft some text).
  (from messages 101, 102)
- gap: as of the 2026-07-28 MCP spec revision (SEP-2577), sampling was formally deprecated —
  along with Roots and Logging — due to low client adoption and the complexity of implementing
  it correctly (human-in-the-loop, model selection, security). It's still fully functional for
  at least a year from that revision, but the guidance going forward is for servers to
  integrate directly with an LLM provider API rather than relying on sampling.
  (from message 102)
- concept: Roots (from message 102)
- concept: Logging (from message 102)
- explore: When an LLM produces text, at each step it outputs a probability distribution over
  possible next tokens, and the actual token is chosen by sampling from that distribution
  (that's what temperature, top-p, and top-k control). So "sampling" is standard shorthand for
  "run inference and get a generated output." (from message 102)

not picked (an explanation, not submitted):

- The 2026-07-28 spec revision (SEP-2577) as a source (reason: no link was given)

#### Example 2: code is not picked

<message id="201" role="user">
how do I make my MCP server call Claude directly instead of using sampling?
</message>
<message id="202" role="assistant">
Install the Anthropic SDK (`pip install anthropic`) and call the Messages API from inside
your tool handler:

    import anthropic

    client = anthropic.AsyncAnthropic()

    async def summarize(text: str) -> str:
        message = await client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Summarize this:\n\n{text}"}],
        )
        return message.content[0].text

Keep the API key in an environment variable on the server, never in the client. Every
parameter is listed in the Messages API reference: https://docs.claude.com/en/api/messages
</message>

subject: Calling Claude from an MCP server
summary: I asked how a server can call Claude directly instead of using sampling. It calls
the provider's SDK in the tool handler and keeps its own API key on the server.
items:

- concept: Keep the API key in an environment variable on the server, never in the client.
  (from message 202)
- source: https://docs.claude.com/en/api/messages (from message 202)

not picked (an explanation, not submitted):

- The code and the install command (reason: details I can look up; the sentence above
  explains the idea)
- The single parameters, such as model and max_tokens (reason: too detailed)
