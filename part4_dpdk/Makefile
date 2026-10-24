# Build the Part 4 receiver against the system DPDK (apt: dpdk-dev, libdpdk-dev).
CFLAGS ?= -O2 -g -Wall -Wextra
PKGCONF = pkg-config

dpdk_rx: dpdk_rx.c
	$(CC) $(CFLAGS) $$($(PKGCONF) --cflags libdpdk) $< -o $@ $$($(PKGCONF) --libs libdpdk)

clean:
	rm -f dpdk_rx

.PHONY: clean
