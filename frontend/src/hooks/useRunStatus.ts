import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

/** Poll a document's run status — fast while a run is active, lazily otherwise. */
export function useRunStatus(docId: string | null) {
  return useQuery({
    queryKey: ['status', docId],
    queryFn: () => api.status(docId!),
    enabled: docId !== null,
    refetchInterval: (query) => (query.state.data?.active ? 400 : 3000),
  })
}
