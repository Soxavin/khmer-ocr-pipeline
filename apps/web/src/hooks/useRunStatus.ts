import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/** Poll a document's run status — fast while a run is active, lazily otherwise. */
export function useRunStatus(docId: string | null) {
  return useQuery({
    queryKey: ['status', docId],
    queryFn: () => api.status(docId!),
    enabled: docId !== null,
    // 400ms while extracting; stop entirely once a doc has settled results (the
    // run-finished invalidation refreshes them); 3s for queued/error docs.
    refetchInterval: (query) =>
      query.state.data?.active ? 400 : query.state.data?.has_results ? false : 3000,
  })
}
