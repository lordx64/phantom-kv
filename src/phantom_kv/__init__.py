"""phantom-kv: refusal removal as a loadable KV-cache graft.

Base weights stay byte-identical; nothing is projected out of activations.
The graft is a learned bank of key/value tensors (or a compiled soft prompt)
spliced into the cache at inference time as phantom context.
"""

__version__ = "0.1.0"
