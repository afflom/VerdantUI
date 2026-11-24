/**
 * Browser localStorage persistence adapter for @assistant-ui/react thread lists
 * and automatic first-prompt thread title generation.
 */

import {
  createLocalStorageAdapter,
  createSimpleTitleAdapter,
  type AsyncStorageLike,
} from '@assistant-ui/core/react';

export function createBrowserAsyncStorage(
  backingStore?: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>,
): AsyncStorageLike {
  const memoryFallback = new Map<string, string>();
  const resolveStore = () => {
    if (backingStore) return backingStore;
    if (typeof window !== 'undefined' && typeof window.localStorage !== 'undefined') {
      return window.localStorage;
    }
    return null;
  };

  return {
    async getItem(key: string): Promise<string | null> {
      const store = resolveStore();
      if (!store) return memoryFallback.get(key) ?? null;
      try {
        return store.getItem(key);
      } catch {
        return memoryFallback.get(key) ?? null;
      }
    },
    async setItem(key: string, value: string): Promise<void> {
      const store = resolveStore();
      if (!store) {
        memoryFallback.set(key, value);
        return;
      }
      try {
        store.setItem(key, value);
      } catch {
        memoryFallback.set(key, value);
      }
    },
    async removeItem(key: string): Promise<void> {
      const store = resolveStore();
      if (!store) {
        memoryFallback.delete(key);
        return;
      }
      try {
        store.removeItem(key);
      } catch {
        memoryFallback.delete(key);
      }
    },
  };
}

export function createVTensorThreadListAdapter(
  backingStore?: Pick<Storage, 'getItem' | 'setItem' | 'removeItem'>,
) {
  return createLocalStorageAdapter({
    storage: createBrowserAsyncStorage(backingStore),
    prefix: 'vtensor-assistant:v1:',
    titleGenerator: createSimpleTitleAdapter(),
  });
}
