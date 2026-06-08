import { resolveAccountRef } from "../utils.js";

/**
 * PswapResource surfaces partial-swap (PSWAP) lineage tracking and creator-side
 * cancellation keyed by a lineage's stable order id.
 *
 * A PSWAP note can be filled by many consumers; each fill advances the lineage
 * to a new remainder note. The client persists that chain locally as it syncs,
 * and the reader methods here expose it. {@link PswapResource#cancelByOrder}
 * reclaims the unfilled offered asset on the current tip of an active lineage,
 * resolving the creator account from the tracked record so the caller only
 * needs the order id.
 */
export class PswapResource {
  #inner;
  #getWasm;
  #client;

  constructor(inner, getWasm, client) {
    this.#inner = inner;
    this.#getWasm = getWasm;
    this.#client = client;
  }

  /**
   * Returns every PSWAP lineage tracked by this client.
   *
   * @returns {Promise<PswapLineageRecord[]>} All tracked lineages.
   */
  async lineages() {
    this.#client.assertNotTerminated();
    return await this.#inner.getPswapLineages();
  }

  /**
   * Returns lineages created by a specific local account.
   *
   * @param {AccountRef} account - Creator account (hex, bech32, Account, or AccountId).
   * @returns {Promise<PswapLineageRecord[]>} Lineages created by the account.
   */
  async lineagesFor(account) {
    this.#client.assertNotTerminated();
    const wasm = await this.#getWasm();
    const accountId = resolveAccountRef(account, wasm);
    return await this.#inner.getPswapLineagesFor(accountId);
  }

  /**
   * Returns the lineage for one order, or null if not tracked.
   *
   * @param {string | number | bigint} orderId - Stable order id (decimal).
   * @returns {Promise<PswapLineageRecord | null>} The lineage, or null.
   */
  async lineage(orderId) {
    this.#client.assertNotTerminated();
    const result = await this.#inner.getPswapLineage(String(orderId));
    return result ?? null;
  }

  /**
   * Builds and submits a transaction reclaiming the unfilled offered asset on
   * the current tip of an Active lineage. The creator account is resolved from
   * the tracked lineage, so only the order id is required. Runs through the
   * same execute/prove/submit path as the other transaction helpers.
   *
   * @param {PswapCancelByOrderOptions} opts - Order id and optional tx options.
   * @returns {Promise<TransactionSubmitResult>} The submitted transaction.
   */
  async cancelByOrder(opts) {
    this.#client.assertNotTerminated();
    const orderId = String(opts.orderId);

    const lineage = await this.#inner.getPswapLineage(orderId);
    if (!lineage) {
      throw new Error(`No PSWAP lineage tracked for order ${orderId}`);
    }
    const accountId = lineage.creatorAccountId();

    const request = await this.#inner.buildPswapCancelByOrder(orderId);

    const { txId, result } = await this.#client.transactions.submit(
      accountId,
      request,
      { prover: opts.prover }
    );

    if (opts.waitForConfirmation) {
      await this.#client.transactions.waitFor(txId.toHex(), {
        timeout: opts.timeout,
      });
    }

    return { txId, result };
  }
}
