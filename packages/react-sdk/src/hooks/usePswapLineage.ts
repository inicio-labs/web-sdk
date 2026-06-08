import { useCallback, useEffect, useState } from "react";
import type { PswapLineageRecord } from "@miden-sdk/miden-sdk";
import { useMiden } from "../context/MidenProvider";
import { useSyncStateStore } from "../store/MidenStore";
import type { PswapLineageResult } from "../types";

/**
 * Hook to fetch a single partial-swap (PSWAP) lineage by its stable order id.
 * Returns `null` if this client is not tracking the order. Refreshes after each
 * successful sync.
 *
 * @param orderId - Stable order id (decimal string or numeric value).
 *
 * @example
 * ```tsx
 * function OrderStatus({ orderId }: { orderId: string }) {
 *   const { lineage, isLoading } = usePswapLineage(orderId);
 *   if (isLoading) return <div>Loading...</div>;
 *   if (!lineage) return <div>Not tracked</div>;
 *   return <div>State: {lineage.state()}</div>;
 * }
 * ```
 */
export function usePswapLineage(
  orderId: string | number | bigint | null | undefined
): PswapLineageResult {
  const { client, isReady } = useMiden();
  const { lastSyncTime } = useSyncStateStore();

  const [lineage, setLineage] = useState<PswapLineageRecord | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const orderIdStr = orderId == null ? null : String(orderId);

  const refetch = useCallback(async () => {
    if (!client || !isReady || orderIdStr == null) return;

    setIsLoading(true);
    setError(null);

    try {
      const fetched = await client.getPswapLineage(orderIdStr);
      setLineage(fetched ?? null);
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setIsLoading(false);
    }
  }, [client, isReady, orderIdStr]);

  useEffect(() => {
    if (!isReady) return;
    refetch();
  }, [isReady, refetch]);

  useEffect(() => {
    if (!isReady || !lastSyncTime) return;
    refetch();
  }, [isReady, lastSyncTime, refetch]);

  return { lineage, isLoading, error, refetch };
}

export type UsePswapLineageResult = PswapLineageResult;
