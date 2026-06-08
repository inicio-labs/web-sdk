import { describe, it, expect, afterEach, vi, beforeEach } from "vitest";
import { openDatabase, getDatabase } from "./schema.js";
import {
  upsertPswapLineage,
  getPswapLineage,
  listPswapLineages,
  applyPswapRound,
} from "./pswap.js";
import { addNoteTag } from "./sync.js";
import { upsertInputNote } from "./notes.js";

// Mirror of the discriminants in pswap.ts (kept private there).
const STATE_ACTIVE = 0;
const STATE_FULLY_FILLED = 1;
const STATE_RECLAIMED = 2;

let dbCounter = 0;
function uniqueDbName(): string {
  return `test-pswap-${++dbCounter}-${Date.now()}`;
}

const openDbIds: string[] = [];

afterEach(async () => {
  for (const dbId of openDbIds) {
    const db = getDatabase(dbId);
    db.dexie.close();
    await db.dexie.delete();
  }
  openDbIds.length = 0;
});

async function openTestDb(): Promise<string> {
  const name = uniqueDbName();
  await openDatabase(name, "0.1.0");
  openDbIds.push(name);
  return name;
}

/** Upsert a lineage row with sensible defaults; override only what a test cares
 *  about. The opaque blobs (originalPswap) are arbitrary — pswap.ts never
 *  deserializes them, only the Rust layer does. */
async function seedLineage(
  dbId: string,
  opts: {
    orderId?: string;
    currentTipNoteId?: string;
    currentDepth?: number;
    remainingOffered?: string;
    remainingRequested?: string;
    state?: number;
    creatorAccountId?: string;
    assetPairTag?: Uint8Array;
    subscriptionAnchorNoteId?: string;
  } = {}
): Promise<void> {
  await upsertPswapLineage(
    dbId,
    opts.orderId ?? "100",
    new Uint8Array([9, 9, 9]),
    opts.currentTipNoteId ?? "0xtip",
    opts.currentDepth ?? 0,
    opts.remainingOffered ?? "1000",
    opts.remainingRequested ?? "500",
    opts.state ?? STATE_ACTIVE,
    10,
    10,
    opts.creatorAccountId ?? "0xcreator",
    opts.assetPairTag ?? new Uint8Array([1, 1]),
    opts.subscriptionAnchorNoteId ?? "0xanchor"
  );
}

/** A `SerializedInputNoteData`-shaped record for round payloads. Returned from a
 *  helper (not an inline literal) so excess-property checks stay relaxed. */
function makeRoundNote(detailsCommitment: string, state: Uint8Array) {
  return {
    detailsCommitment,
    noteId: detailsCommitment,
    noteAssets: new Uint8Array([1, 2, 3]),
    attachments: new Uint8Array([4]),
    serialNumber: new Uint8Array([5]),
    inputs: new Uint8Array([6]),
    noteScriptRoot: `root-${detailsCommitment}`,
    noteScript: new Uint8Array([7]),
    nullifier: `null-${detailsCommitment}`,
    createdAt: "0",
    stateDiscriminant: 0,
    state,
  };
}

describe("pswap", () => {
  let errorSpy: any;
  let logSpy: any;

  beforeEach(() => {
    errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
    logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
  });

  afterEach(() => {
    errorSpy.mockRestore();
    logSpy.mockRestore();
  });

  // upsert + get
  // ------------------------------------------------------------------------

  it("upserts and retrieves a lineage, preserving u64 amounts beyond 2^53", async () => {
    const dbId = await openTestDb();
    await upsertPswapLineage(
      dbId,
      "42",
      new Uint8Array([1, 2, 3]),
      "0xtip",
      0,
      // u64::MAX — would lose precision if round-tripped through a JS number.
      "18446744073709551615",
      "500",
      STATE_ACTIVE,
      10,
      12,
      "0xcreator",
      new Uint8Array([7, 7]),
      "0xanchor"
    );

    const row = await getPswapLineage(dbId, "42");
    expect(row).not.toBeNull();
    expect(row!.orderId).toBe("42");
    expect(row!.currentTipNoteId).toBe("0xtip");
    expect(row!.currentDepth).toBe(0);
    expect(row!.remainingOffered).toBe("18446744073709551615");
    expect(row!.remainingRequested).toBe("500");
    expect(row!.state).toBe(STATE_ACTIVE);
    expect(row!.createdAtBlock).toBe(10);
    expect(row!.updatedAtBlock).toBe(12);
    expect(row!.creatorAccountId).toBe("0xcreator");
    expect(row!.subscriptionAnchorNoteId).toBe("0xanchor");
    // Blobs are base64-encoded on the way in.
    expect(row!.originalPswap).toBe("AQID");
    expect(row!.assetPairTag).toBe("Bwc=");
  });

  it("upsert overwrites an existing lineage by orderId", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, {
      orderId: "7",
      currentDepth: 0,
      state: STATE_ACTIVE,
    });
    await upsertPswapLineage(
      dbId,
      "7",
      new Uint8Array([1]),
      "0xnewtip",
      3,
      "1",
      "2",
      STATE_FULLY_FILLED,
      1,
      9,
      "0xc",
      new Uint8Array([2]),
      "0xa"
    );

    const row = await getPswapLineage(dbId, "7");
    expect(row!.currentTipNoteId).toBe("0xnewtip");
    expect(row!.currentDepth).toBe(3);
    expect(row!.state).toBe(STATE_FULLY_FILLED);
  });

  it("getPswapLineage returns null for an unknown orderId", async () => {
    const dbId = await openTestDb();
    expect(await getPswapLineage(dbId, "999")).toBeNull();
  });

  // list filters
  // ------------------------------------------------------------------------

  it("lists only Active lineages with the Active filter", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, { orderId: "1", state: STATE_ACTIVE });
    await seedLineage(dbId, { orderId: "2", state: STATE_FULLY_FILLED });
    await seedLineage(dbId, { orderId: "3", state: STATE_ACTIVE });

    const rows = await listPswapLineages(dbId, "Active");
    expect(rows!.map((r) => r.orderId).sort()).toEqual(["1", "3"]);
  });

  it("lists lineages by creator, returning [] when none match", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, { orderId: "1", creatorAccountId: "0xalice" });
    await seedLineage(dbId, { orderId: "2", creatorAccountId: "0xbob" });

    const mine = await listPswapLineages(dbId, "ByCreator:0xalice");
    expect(mine!.map((r) => r.orderId)).toEqual(["1"]);

    const none = await listPswapLineages(dbId, "ByCreator:0xnobody");
    expect(none).toEqual([]);
  });

  it("lists active lineages by tip note ids, excluding non-active ones", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, {
      orderId: "1",
      currentTipNoteId: "0xaaa",
      state: STATE_ACTIVE,
    });
    await seedLineage(dbId, {
      orderId: "2",
      currentTipNoteId: "0xbbb",
      state: STATE_RECLAIMED,
    });
    await seedLineage(dbId, {
      orderId: "3",
      currentTipNoteId: "0xccc",
      state: STATE_ACTIVE,
    });

    // 0xbbb is reclaimed → filtered out; 0xccc not requested.
    const rows = await listPswapLineages(
      dbId,
      "ActiveByTipNoteIds:0xaaa,0xbbb"
    );
    expect(rows!.map((r) => r.orderId)).toEqual(["1"]);
  });

  it("returns [] for ActiveByTipNoteIds with no ids", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, {
      orderId: "1",
      currentTipNoteId: "0xaaa",
      state: STATE_ACTIVE,
    });
    expect(await listPswapLineages(dbId, "ActiveByTipNoteIds:")).toEqual([]);
  });

  it("lists every lineage with the All filter", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, { orderId: "1", state: STATE_ACTIVE });
    await seedLineage(dbId, { orderId: "2", state: STATE_RECLAIMED });

    const rows = await listPswapLineages(dbId, "All");
    expect(rows!.length).toBe(2);
  });

  it("treats an unrecognized filter as All", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, { orderId: "1" });

    const rows = await listPswapLineages(dbId, "SomethingElse");
    expect(rows!.length).toBe(1);
  });

  // applyPswapRound
  // ------------------------------------------------------------------------

  it("applies a non-terminal round: advances the lineage, inserts notes, keeps the tag", async () => {
    const dbId = await openTestDb();
    const tagBytes = new Uint8Array([5, 5]);
    const anchor = "0xanchor-active";
    await seedLineage(dbId, {
      orderId: "5",
      currentDepth: 0,
      currentTipNoteId: "0xold",
      state: STATE_ACTIVE,
      assetPairTag: tagBytes,
      subscriptionAnchorNoteId: anchor,
    });
    await addNoteTag(dbId, tagBytes, "", "", anchor);

    await applyPswapRound(dbId, {
      orderId: "5",
      roundDepth: 1,
      tipNoteId: "0xnew",
      remainingOffered: "900",
      remainingRequested: "450",
      state: STATE_ACTIVE,
      updatedAtBlock: 20,
      inputNotes: [
        makeRoundNote("dc-pay", new Uint8Array([1])),
        makeRoundNote("dc-rem", new Uint8Array([2])),
      ],
    });

    const row = await getPswapLineage(dbId, "5");
    expect(row!.currentDepth).toBe(1);
    expect(row!.currentTipNoteId).toBe("0xnew");
    expect(row!.remainingOffered).toBe("900");
    expect(row!.remainingRequested).toBe("450");
    expect(row!.state).toBe(STATE_ACTIVE);
    expect(row!.updatedAtBlock).toBe(20);

    const db = getDatabase(dbId);
    expect(await db.inputNotes.get("dc-pay")).toBeDefined();
    expect(await db.inputNotes.get("dc-rem")).toBeDefined();
    // Non-terminal: the subscription tag survives.
    expect(await db.tags.count()).toBe(1);
  });

  it("drops the subscription tag and freezes the tip on a FullyFilled round", async () => {
    const dbId = await openTestDb();
    const tagBytes = new Uint8Array([3, 1, 4]);
    const anchor = "0xanchor-ff";
    await seedLineage(dbId, {
      orderId: "8",
      currentDepth: 0,
      currentTipNoteId: "0xfrozen",
      state: STATE_ACTIVE,
      assetPairTag: tagBytes,
      subscriptionAnchorNoteId: anchor,
    });
    await addNoteTag(dbId, tagBytes, "", "", anchor);

    const db = getDatabase(dbId);
    expect(await db.tags.count()).toBe(1);

    await applyPswapRound(dbId, {
      orderId: "8",
      roundDepth: 1,
      // Terminal round carries no new tip.
      tipNoteId: undefined,
      remainingOffered: "0",
      remainingRequested: "0",
      state: STATE_FULLY_FILLED,
      updatedAtBlock: 30,
      inputNotes: [],
    });

    expect(await db.tags.count()).toBe(0);
    const row = await getPswapLineage(dbId, "8");
    expect(row!.state).toBe(STATE_FULLY_FILLED);
    // Frozen tip: the previous tip is retained.
    expect(row!.currentTipNoteId).toBe("0xfrozen");
  });

  it("drops the subscription tag on a Reclaimed round", async () => {
    const dbId = await openTestDb();
    const tagBytes = new Uint8Array([2, 7, 1, 8]);
    const anchor = "0xanchor-rc";
    await seedLineage(dbId, {
      orderId: "12",
      currentDepth: 0,
      state: STATE_ACTIVE,
      assetPairTag: tagBytes,
      subscriptionAnchorNoteId: anchor,
    });
    await addNoteTag(dbId, tagBytes, "", "", anchor);

    const db = getDatabase(dbId);
    expect(await db.tags.count()).toBe(1);

    await applyPswapRound(dbId, {
      orderId: "12",
      roundDepth: 1,
      tipNoteId: undefined,
      remainingOffered: "0",
      remainingRequested: "0",
      state: STATE_RECLAIMED,
      updatedAtBlock: 33,
      inputNotes: [],
    });

    expect(await db.tags.count()).toBe(0);
    expect((await getPswapLineage(dbId, "12"))!.state).toBe(STATE_RECLAIMED);
  });

  it("skips an input note that already exists, leaving its state intact", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, {
      orderId: "9",
      currentDepth: 0,
      state: STATE_ACTIVE,
    });

    const originalState = new Uint8Array([10, 20, 30]);
    // Simulate the screener having pre-inserted the public payback.
    await upsertInputNote(
      dbId,
      "dc-existing",
      "dc-existing",
      new Uint8Array([0]),
      new Uint8Array([0]),
      new Uint8Array([0]),
      new Uint8Array([0]),
      "root-x",
      new Uint8Array([0]),
      "null-x",
      "0",
      0,
      originalState
    );

    await applyPswapRound(dbId, {
      orderId: "9",
      roundDepth: 1,
      tipNoteId: "0xnew",
      remainingOffered: "0",
      remainingRequested: "0",
      state: STATE_ACTIVE,
      updatedAtBlock: 40,
      // Same details commitment, different state — must NOT clobber.
      inputNotes: [makeRoundNote("dc-existing", new Uint8Array([99, 99]))],
    });

    const db = getDatabase(dbId);
    const stored = await db.inputNotes.get("dc-existing");
    expect(Array.from(stored!.state)).toEqual([10, 20, 30]);
  });

  it("throws on an unknown orderId", async () => {
    const dbId = await openTestDb();
    await expect(
      applyPswapRound(dbId, {
        orderId: "404",
        roundDepth: 1,
        tipNoteId: "0xnew",
        remainingOffered: "0",
        remainingRequested: "0",
        state: STATE_ACTIVE,
        updatedAtBlock: 1,
        inputNotes: [],
      })
    ).rejects.toThrow(/unknown order_id 404/);
  });

  it("throws on a non-monotonic round depth and rolls back the whole round", async () => {
    const dbId = await openTestDb();
    await seedLineage(dbId, {
      orderId: "11",
      currentDepth: 0,
      currentTipNoteId: "0xold",
      state: STATE_ACTIVE,
    });

    await expect(
      applyPswapRound(dbId, {
        orderId: "11",
        roundDepth: 5,
        tipNoteId: "0xnew",
        remainingOffered: "1",
        remainingRequested: "1",
        state: STATE_ACTIVE,
        updatedAtBlock: 99,
        inputNotes: [makeRoundNote("dc-should-not-exist", new Uint8Array([1]))],
      })
    ).rejects.toThrow(/does not advance current_depth/);

    // Rollback: lineage untouched and the round's note never landed.
    const row = await getPswapLineage(dbId, "11");
    expect(row!.currentDepth).toBe(0);
    expect(row!.currentTipNoteId).toBe("0xold");
    const db = getDatabase(dbId);
    expect(await db.inputNotes.get("dc-should-not-exist")).toBeUndefined();
  });

  // error propagation (logWebStoreError re-throws)
  // ------------------------------------------------------------------------

  it("upsertPswapLineage rejects on a Dexie error", async () => {
    await expect(
      upsertPswapLineage(
        "never-opened",
        "1",
        new Uint8Array([1]),
        "0xtip",
        0,
        "1",
        "1",
        STATE_ACTIVE,
        0,
        0,
        "0xc",
        new Uint8Array([1]),
        "0xa"
      )
    ).rejects.toThrow();
  });

  it("getPswapLineage rejects on a Dexie error", async () => {
    await expect(getPswapLineage("never-opened", "1")).rejects.toThrow();
  });

  it("listPswapLineages rejects on a Dexie error", async () => {
    await expect(listPswapLineages("never-opened", "All")).rejects.toThrow();
  });

  it("applyPswapRound rejects when the database is not open", async () => {
    await expect(
      applyPswapRound("never-opened", {
        orderId: "1",
        roundDepth: 1,
        tipNoteId: "0xnew",
        remainingOffered: "0",
        remainingRequested: "0",
        state: STATE_ACTIVE,
        updatedAtBlock: 1,
        inputNotes: [],
      })
    ).rejects.toThrow();
  });
});
