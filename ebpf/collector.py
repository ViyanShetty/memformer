#!/usr/bin/env python3
from bcc import BPF
import sys, ctypes

prog = r"""
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 pc;
    u64 addr;
    u32 pid;
};

BPF_PERF_OUTPUT(events);

int trace_page_fault(struct pt_regs *ctx, unsigned long address, unsigned int flags) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct event_t e = {};
    e.pid = pid;
    e.pc  = PT_REGS_IP(ctx);
    e.addr = address;
    events.perf_submit(ctx, &e, sizeof(e));
    return 0;
}
"""

class Event(ctypes.Structure):
    _fields_ = [("pc", ctypes.c_uint64),
                ("addr", ctypes.c_uint64),
                ("pid", ctypes.c_uint32)]

b = BPF(text=prog)
b.attach_kprobe(event="handle_mm_fault", fn_name="trace_page_fault")

count = 0
outfile = open("traces/ebpf_live.out", "w")

def handle_event(cpu, data, size):
    global count
    e = b["events"].event(data)
    outfile.write(f"{e.pc:x} {e.addr:x}\n")
    count += 1
    if count % 5000 == 0:
        print(f"Collected {count} events...", flush=True)
    if count >= 50000:
        outfile.close()
        print(f"Done. Saved {count} events to traces/ebpf_live.out")
        exit(0)

b["events"].open_perf_buffer(handle_event)
print("Tracing all page faults... Ctrl-C to stop early")
while True:
    b.perf_buffer_poll()