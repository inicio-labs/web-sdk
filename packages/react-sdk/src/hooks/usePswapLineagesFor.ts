import { useCallback, useEffect, useState } from "react";
import type { PswapLineageRecord } from "@miden-sdk/miden-sdk";
import { useMiden } from "../context/MidenProvider";
import { useSyncStateStore } from "../store/MidenStore";
import type { AccountRef, PswapLineagesResult } from "../types";
import { parseAccountId } from "../utils/accountParsing";

/**
 * Hook to list the partial-swap (PSWAP) lineages created by a specific local
 * account. Refreshes after each successful sync.
 *
 * @param account - Creator account (hex, bech32, `Account`, or `AccountId`).
 *
 * @example
 * ```tsx
 * function MyLineages({ accountId }: { accountId: string }) {
 *   const { lineages, isLoading } = usePswapLineagesFor(accountId);
 *   if (isLoading) return <div>Loading...</div>;
 *   return <div>{lineages.length} open orders</div>;
 * }
 * ```
 */
export function usePswapLineagesFor(
  account: AccountRef | null | undefined
): PswapLineagesResult {
  const { client, isReady } = useMiden();
  const { lastSyncTime } = useSyncStateStore();

  const [lineages, setLineages] = useState<PswapLineageRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!client || !isReady || account == null) return;

    setIsLoading(true);
    setError(null);

    try {
      const accountIdObj = parseAccountId(account);
      const fetched = await client.getPswapLineagesFor(accountIdObj);
      setLineages(fetched);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, [client, isReady, account]);

  useEffect(() => {
    if (!isReady) return;
    refetch();
  }, [isReady, refetch]);

  useEffect(() => {
    if (!isReady || !lastSyncTime) return;
    refetch();
  }, [isReady, lastSyncTime, refetch]);

  return { lineages, isLoading, error, refetch };
}

export type UsePswapLineagesForResult = PswapLineagesResult;
