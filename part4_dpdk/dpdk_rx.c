/*
 * Part 4: receive the Part 3 image packets with DPDK and rebuild the images.
 *
 *   ./dpdk_rx <EAL args> -- OUTPUT [--expect N] [--idle-ms MS]
 *
 *   sudo ./dpdk_rx -l 0 --no-huge -m 512 --no-pci \
 *       --vdev 'net_pcap0,rx_pcap=images.pcap,tx_pcap=/dev/null' -- frames_rx.bin
 *
 * Every packet is checked layer by layer (Ethernet -> IPv4 -> UDP -> our 22-byte header)
 * with a bounds check before each read; anything else is counted and dropped. Chunks are
 * copied to their offset, so arrival order does not matter, and a per-frame bitmap of
 * received sequence numbers drops duplicates.
 *
 * End of input: the pcap port never reports end of file, it just keeps returning empty
 * bursts. The receiver stops when --expect N frames are complete, or when no packet has
 * arrived for --idle-ms (default 200 ms) after the first one.
 *
 * OUTPUT gets the complete frames in frame_id order, headerless, FRAME_BYTES each -- the
 * layout of Part 3's original.bin -- and OUTPUT.ids.json lists the frame IDs written, so
 * Part 5 can match frames even if one is missing.
 *
 * Packet format: part3_packets/frame_format.py (all fields big-endian).
 */
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <rte_byteorder.h>
#include <rte_cycles.h>
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_ether.h>
#include <rte_ip.h>
#include <rte_mbuf.h>
#include <rte_udp.h>

/* Must match part3_packets/frame_format.py */
#define MAGIC       0x45453538u
#define UDP_PORT    5000
#define HEADER_LEN  22
#define FRAME_BYTES (224u * 224u * 3u)
#define CHUNK       1400u
#define MAX_FRAMES  1024u

#define PORT        0
#define BURST       32
#define NUM_MBUFS   8191
#define MBUF_CACHE  256
#define RX_RING     1024
#define TX_RING     1024

struct frame {
	uint8_t *data;      /* FRAME_BYTES, allocated on the first chunk */
	uint8_t *got;       /* got[seq] = 1 once that chunk has been copied */
	uint16_t nseg;
	uint16_t received;
};

struct stats {
	uint64_t rx, accepted, duplicates;
	uint64_t drop_not_ipv4, drop_bad_ip, drop_not_udp, drop_other_port;
	uint64_t drop_short, drop_bad_magic, drop_bad_header, drop_multiseg;
};

static struct frame frames[MAX_FRAMES];
static struct stats st;

/* Our header, read field by field with memcpy: the payload has no alignment guarantee. */
struct chunk_hdr {
	uint32_t magic, frame_id;
	uint16_t seq, nseg;
	uint32_t offset;
	uint16_t length;
	uint32_t frame_bytes;
};

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return rte_be_to_cpu_32(v); }
static uint16_t rd16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return rte_be_to_cpu_16(v); }

static void parse_chunk_hdr(const uint8_t *p, struct chunk_hdr *h)
{
	h->magic = rd32(p + 0);
	h->frame_id = rd32(p + 4);
	h->seq = rd16(p + 8);
	h->nseg = rd16(p + 10);
	h->offset = rd32(p + 12);
	h->length = rd16(p + 16);
	h->frame_bytes = rd32(p + 18);
}

/* Validate one packet and copy its chunk into place. */
static void handle_packet(struct rte_mbuf *m)
{
	st.rx++;
	if (m->nb_segs != 1) {              /* our packets always fit one mbuf */
		st.drop_multiseg++;
		return;
	}
	const uint8_t *pkt = rte_pktmbuf_mtod(m, const uint8_t *);
	uint32_t len = rte_pktmbuf_data_len(m);

	/* Ethernet */
	if (len < sizeof(struct rte_ether_hdr)) { st.drop_not_ipv4++; return; }
	const struct rte_ether_hdr *eth = (const struct rte_ether_hdr *)pkt;
	if (eth->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) { st.drop_not_ipv4++; return; }
	uint32_t pos = sizeof(*eth);

	/* IPv4: version, header length from IHL, total length, no fragments, checksum */
	if (len - pos < sizeof(struct rte_ipv4_hdr)) { st.drop_bad_ip++; return; }
	const struct rte_ipv4_hdr *ip = (const struct rte_ipv4_hdr *)(pkt + pos);
	uint32_t ihl = (ip->version_ihl & 0x0f) * 4u;
	uint32_t ip_total = rte_be_to_cpu_16(ip->total_length);
	uint16_t frag = rte_be_to_cpu_16(ip->fragment_offset);
	if ((ip->version_ihl >> 4) != 4 || ihl < sizeof(*ip) || ip_total < ihl || ip_total > len - pos ||
	    (frag & (RTE_IPV4_HDR_MF_FLAG | RTE_IPV4_HDR_OFFSET_MASK)) != 0 ||
	    rte_raw_cksum(ip, ihl) != 0xffff) {
		st.drop_bad_ip++;
		return;
	}
	if (ip->next_proto_id != IPPROTO_UDP) { st.drop_not_udp++; return; }
	uint32_t ip_end = pos + ip_total;    /* ignore any Ethernet padding after the IP datagram */
	pos += ihl;

	/* UDP */
	if (ip_end - pos < sizeof(struct rte_udp_hdr)) { st.drop_bad_ip++; return; }
	const struct rte_udp_hdr *udp = (const struct rte_udp_hdr *)(pkt + pos);
	if (udp->dst_port != rte_cpu_to_be_16(UDP_PORT)) { st.drop_other_port++; return; }
	uint32_t udp_len = rte_be_to_cpu_16(udp->dgram_len);
	if (udp_len < sizeof(*udp) || udp_len > ip_end - pos) { st.drop_bad_ip++; return; }
	pos += sizeof(*udp);
	uint32_t payload_len = udp_len - sizeof(*udp);

	/* Our header */
	if (payload_len < HEADER_LEN) { st.drop_short++; return; }
	struct chunk_hdr h;
	parse_chunk_hdr(pkt + pos, &h);
	if (h.magic != MAGIC) { st.drop_bad_magic++; return; }
	if (h.frame_id >= MAX_FRAMES || h.frame_bytes != FRAME_BYTES ||
	    h.nseg != (FRAME_BYTES + CHUNK - 1) / CHUNK || h.seq >= h.nseg ||
	    h.length != payload_len - HEADER_LEN || h.offset != (uint32_t)h.seq * CHUNK ||
	    h.offset + h.length > h.frame_bytes) {
		st.drop_bad_header++;
		return;
	}

	struct frame *f = &frames[h.frame_id];
	if (f->data == NULL) {
		f->data = calloc(1, FRAME_BYTES);
		f->got = calloc(h.nseg, 1);
		if (f->data == NULL || f->got == NULL)
			rte_exit(EXIT_FAILURE, "out of memory\n");
		f->nseg = h.nseg;
	}
	if (f->got[h.seq]) {                 /* already have this chunk */
		st.duplicates++;
		return;
	}
	memcpy(f->data + h.offset, pkt + pos + HEADER_LEN, h.length);
	f->got[h.seq] = 1;
	f->received++;
	st.accepted++;
}

static unsigned count_complete(void)
{
	unsigned n = 0;
	for (unsigned i = 0; i < MAX_FRAMES; i++)
		if (frames[i].data != NULL && frames[i].received == frames[i].nseg)
			n++;
	return n;
}

static int write_output(const char *path)
{
	char ids_path[4096];
	snprintf(ids_path, sizeof(ids_path), "%s.ids.json", path);
	FILE *out = fopen(path, "wb"), *ids = fopen(ids_path, "w");
	if (out == NULL || ids == NULL) {
		fprintf(stderr, "cannot open %s: %s\n", out == NULL ? path : ids_path, strerror(errno));
		return -1;
	}
	fputc('[', ids);
	bool first = true;
	for (unsigned i = 0; i < MAX_FRAMES; i++) {
		struct frame *f = &frames[i];
		if (f->data == NULL)
			continue;
		if (f->received != f->nseg) {
			printf("frame %u: incomplete, %u/%u chunks; missing seq", i, f->received, f->nseg);
			unsigned shown = 0;
			for (unsigned s = 0; s < f->nseg && shown < 10; s++)
				if (!f->got[s]) { printf(" %u", s); shown++; }
			printf("%s\n", (unsigned)(f->nseg - f->received) > shown ? " ..." : "");
			continue;
		}
		if (fwrite(f->data, 1, FRAME_BYTES, out) != FRAME_BYTES) {
			fprintf(stderr, "write %s failed\n", path);
			return -1;
		}
		fprintf(ids, "%s%u", first ? "" : ", ", i);
		first = false;
	}
	fputs("]\n", ids);
	fclose(out);
	fclose(ids);
	return 0;
}

static void usage(const char *prog)
{
	fprintf(stderr, "usage: %s <EAL args> -- OUTPUT [--expect N] [--idle-ms MS]\n", prog);
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	int ret = rte_eal_init(argc, argv);
	if (ret < 0)
		rte_exit(EXIT_FAILURE, "EAL init failed\n");
	argc -= ret;                         /* argv[0] is now the program name */
	argv += ret;

	const char *output = NULL;
	unsigned expect = 0, idle_ms = 200;
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "--expect") == 0 && i + 1 < argc)
			expect = (unsigned)strtoul(argv[++i], NULL, 10);
		else if (strcmp(argv[i], "--idle-ms") == 0 && i + 1 < argc)
			idle_ms = (unsigned)strtoul(argv[++i], NULL, 10);
		else if (output == NULL && argv[i][0] != '-')
			output = argv[i];
		else
			usage(argv[0]);
	}
	if (output == NULL)
		usage(argv[0]);
	if (rte_eth_dev_count_avail() == 0)
		rte_exit(EXIT_FAILURE, "no port: pass --vdev 'net_pcap0,rx_pcap=FILE,tx_pcap=/dev/null'\n");

	struct rte_mempool *pool = rte_pktmbuf_pool_create("mbufs", NUM_MBUFS, MBUF_CACHE, 0,
							   RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
	if (pool == NULL)
		rte_exit(EXIT_FAILURE, "mbuf pool: %s\n", rte_strerror(rte_errno));
	struct rte_eth_conf conf;
	memset(&conf, 0, sizeof(conf));
	if (rte_eth_dev_configure(PORT, 1, 1, &conf) < 0 ||
	    rte_eth_rx_queue_setup(PORT, 0, RX_RING, rte_eth_dev_socket_id(PORT), NULL, pool) < 0 ||
	    rte_eth_tx_queue_setup(PORT, 0, TX_RING, rte_eth_dev_socket_id(PORT), NULL) < 0 ||
	    rte_eth_dev_start(PORT) < 0)
		rte_exit(EXIT_FAILURE, "port %d setup failed\n", PORT);

	const uint64_t hz = rte_get_timer_hz();
	const uint64_t idle_cycles = hz * idle_ms / 1000;
	const uint64_t start = rte_get_timer_cycles();
	uint64_t first_rx = 0, last_rx = 0;
	struct rte_mbuf *bufs[BURST];
	const char *reason = "idle";

	for (;;) {
		uint16_t n = rte_eth_rx_burst(PORT, 0, bufs, BURST);
		uint64_t now = rte_get_timer_cycles();
		if (n > 0) {
			if (first_rx == 0)
				first_rx = now;
			last_rx = now;
			for (uint16_t i = 0; i < n; i++) {
				handle_packet(bufs[i]);
				rte_pktmbuf_free(bufs[i]);   /* every mbuf, accepted or dropped */
			}
			if (expect > 0 && count_complete() >= expect) {
				reason = "expected frames complete";
				break;
			}
		} else if (first_rx != 0 && now - last_rx > idle_cycles) {
			break;                           /* input has gone quiet */
		} else if (first_rx == 0 && now - start > 10 * hz) {
			reason = "no packets within 10 s";
			break;
		}
	}

	double active_ms = first_rx ? (double)(last_rx - first_rx) * 1000.0 / hz : 0.0;
	unsigned complete = count_complete(), seen = 0;
	for (unsigned i = 0; i < MAX_FRAMES; i++)
		seen += frames[i].data != NULL;

	int rc = write_output(output);
	printf("stopped: %s (idle timeout %u ms)\n", reason, idle_ms);
	printf("rx=%" PRIu64 " accepted=%" PRIu64 " duplicates=%" PRIu64 " drop_other_port=%" PRIu64
	       " drop_not_ipv4=%" PRIu64 " drop_bad_ip=%" PRIu64 " drop_not_udp=%" PRIu64
	       " drop_short=%" PRIu64 " drop_bad_magic=%" PRIu64 " drop_bad_header=%" PRIu64
	       " drop_multiseg=%" PRIu64 "\n",
	       st.rx, st.accepted, st.duplicates, st.drop_other_port, st.drop_not_ipv4, st.drop_bad_ip,
	       st.drop_not_udp, st.drop_short, st.drop_bad_magic, st.drop_bad_header, st.drop_multiseg);
	printf("frames: %u complete of %u seen -> %s (+ .ids.json); receive window %.3f ms\n",
	       complete, seen, output, active_ms);

	for (unsigned i = 0; i < MAX_FRAMES; i++) {
		free(frames[i].data);
		free(frames[i].got);
	}
	rte_eth_dev_stop(PORT);
	rte_eth_dev_close(PORT);
	rte_eal_cleanup();
	if (rc != 0 || complete != seen || (expect > 0 && complete < expect))
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}
