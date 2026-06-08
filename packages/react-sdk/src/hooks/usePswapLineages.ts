import { useCallback, useEffect, useState } from "react";
import type { PswapLineageRecord } from "@miden-sdk/miden-sdk";
import { useMiden } from "../context/MidenProvider";
import { useSyncStateStore } from "../store/MidenStore";
import type { PswapLineagesResult } from "../types";

/**
 * Hook to list every partial-swap (PSWAP) lineage tracked by this client.
 *
 * A lineage records how a PSWAP note has been filled round by round, from the
 * original note through each remainder to the current tip. The list refreshes
 * after each successful sync.
 *
 * @example
 * ```tsx
 * function LineageList() {
 *   const { lineages, isLoading } = usePswapLineages();
 *   if (isLoading) return <div>Loading...</div>;
 *   return (
 *     <ul>
 *       {lineages.map((l) => (
 *         <li key={l.orderId()}>
 *           {l.orderId()} — {l.remainingOffered().toString()} remaining
 *         </li>
 *       ))}
 *     </ul>
 *   );
 * }
 * ```
 */
export function usePswapLineages(): PswapLineagesResult {
  const { client, isReady } = useMiden();
  const { lastSyncTime } = useSyncStateStore();

  const [lineages, setLineages] = useState<PswapLineageRecord[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const refetch = useCallback(async () => {
    if (!client || !isReady) return;

    setIsLoading(true);
    setError(null);

    try {
      const fetched = await client.getPswapLineages();
      setLineages(fetched);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, [client, isReady]);

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

export type UsePswapLineagesResult = PswapLineagesResult;
