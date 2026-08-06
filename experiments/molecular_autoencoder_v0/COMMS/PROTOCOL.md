# COMMS protocol

Direction channel between the planning agent (writes `INBOX.md`) and the GPU
agent (writes `ACK.md` and reports via commit messages). Replaces manual
copy-paste, which silently dropped at least two instruction sets.

## Standing rules for the GPU agent

1. **After every push, read `INBOX.md`.** Act on every item whose number is
   greater than `last_acted` in `ACK.md`.
2. **Acknowledge before acting.** Append to `ACK.md`: item number, one-line
   restatement of what you understood, and `ACCEPTED` / `QUESTIONED`.
   Restating in your own words is the check that the item arrived intact.
3. **Push `ACK.md` with your next commit.** An item with no ACK is an item
   that did not arrive — that is the failure this channel exists to prevent.
4. **Disagree in `ACK.md`, don't silently skip.** `QUESTIONED` with a reason
   is a valid response. Silence is not.
5. **Reports stay in commit messages**, as now. The planning agent reads them
   directly from git history; do not duplicate them here.

## Standing rules for the planning agent

1. Number every item monotonically. Never reuse or renumber.
2. One item = one decision or one experiment. Do not bundle.
3. State the pre-registered read before results exist, not after.
4. Do not restate items already ACKed unless the ACK shows a misread.

## Numbering

Items are `NNN` zero-padded, appended at the end of `INBOX.md`, newest last.
`ACK.md` is append-only.
