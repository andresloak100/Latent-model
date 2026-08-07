#!/usr/bin/env python3
"""INBOX 24a: make ACK-ledger divergence SELF-DETECTING.

WHY THIS EXISTS. `ACK.md` sat at `last_acted: 019` while items 020, 021, 022 and 023 were received
and acted on. The work was delivered; the RECORD failed, silently, for four consecutive rounds.

The mechanism was mundane and worth naming: item 020 never got a ledger entry, and every later edit
then ran `s.replace("last_acted: 020", "last_acted: 021")` against a file that still said `019`.
`str.replace` on an absent pattern does nothing and raises nothing. Four no-ops. Every other edit in
that session used `assert old in s` and would have failed loudly; the ledger edits did not.

THE POINT OF THE PROTOCOL IS THAT SILENCE IS DETECTABLE. A commit message asserting "ACKed" is not an
ACK -- it is a claim about the record, in the same place the record was supposed to be checked. A
guard that can be satisfied by prose is not a guard, which is the same defect as a coverage check
that passes a truncated sample because it only compares medians.

Run standalone, or as a pre-push hook:
    ln -sf ../../experiments/molecular_autoencoder_v0/COMMS/check_ack.py .git/hooks/pre-push
Exit code 1 blocks the push.
"""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
INBOX = os.path.join(HERE, "INBOX.md")
ACK = os.path.join(HERE, "ACK.md")


def main():
    if not (os.path.exists(INBOX) and os.path.exists(ACK)):
        print("[check_ack] INBOX.md or ACK.md missing", file=sys.stderr)
        return 1
    inbox = open(INBOX).read()
    ack = open(ACK).read()

    items = sorted({int(m) for m in re.findall(r"^##\s+(\d{3})\b", inbox, re.M)})
    if not items:
        print("[check_ack] no items found in INBOX.md", file=sys.stderr)
        return 1
    highest = items[-1]

    m = re.search(r"^last_acted:\s*(\d+)", ack, re.M)
    if not m:
        print("[check_ack] ACK.md has no `last_acted:` line", file=sys.stderr)
        return 1
    last = int(m.group(1))

    rows = {int(x) for x in re.findall(r"^\|\s*(\d{3})\s*\|", ack, re.M)}
    missing = [i for i in items if i not in rows]

    bad = False
    if last != highest:
        print(f"[check_ack] FAIL: last_acted is {last:03d} but INBOX.md's highest item is "
              f"{highest:03d}. Either act on the gap or say why in ACK.md -- silence is not a valid "
              f"response (PROTOCOL.md).", file=sys.stderr)
        bad = True
    if missing:
        print(f"[check_ack] FAIL: no ACK table row for item(s) "
              f"{', '.join(f'{i:03d}' for i in missing)}. A commit message asserting 'ACKed' is not "
              f"an ACK; the ledger is the artifact.", file=sys.stderr)
        bad = True
    if bad:
        print(f"[check_ack] (items in INBOX: {items[0]:03d}-{highest:03d}; rows in ACK: {len(rows)})",
              file=sys.stderr)
        return 1
    print(f"[check_ack] OK: last_acted {last:03d} == highest INBOX item, {len(rows)} rows, none missing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
