# Quote-and-chunk citation proposal

**Proposal only — not implemented.**

Instead of retrieving the top semantic matches after extraction, the model could
return a quote and chunk ID alongside each non-null field. The application would
look up that chunk in the exact set supplied to the model and verify that the
returned quote is an exact substring of its text. A verified quote could then
be tied to the chunk's page metadata, giving a traceable passage directly
associated with the extracted field.

If the chunk ID is unknown, the quote is absent from that chunk, or the field has
no quote, the system should mark its citation as unverified and require human
review rather than falling back to a post-hoc top-k result as if it were direct
support. This check can establish that the quoted text occurs in the cited
chunk; it cannot establish that the quote entails the field, that the quote is
complete, or that the extraction is factually correct. Human spot-checking
would still be required.
