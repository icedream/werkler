package ai

import (
	"context"
	"fmt"
	"sync"
)

// MultiClient aggregates multiple AI provider clients and routes requests to
// the currently active provider. It implements StreamCompleter, Completer, and
// ModelManager.
type MultiClient struct {
	entries []*multiEntry
	active  int // index into entries
	mu      sync.RWMutex
}

type multiEntry struct {
	providerName string
	sc           StreamCompleter
	completer    Completer    // optional; nil if not supported
	mm           ModelManager // optional; nil if not supported
}

// AddProvider registers a named provider. sc must implement StreamCompleter;
// it will automatically be used as Completer and/or ModelManager if it also
// implements those interfaces.
func (mc *MultiClient) AddProvider(name string, sc StreamCompleter) {
	e := &multiEntry{
		providerName: name,
		sc:           sc,
	}
	if c, ok := sc.(Completer); ok {
		e.completer = c
	}
	if m, ok := sc.(ModelManager); ok {
		e.mm = m
	}
	mc.mu.Lock()
	mc.entries = append(mc.entries, e)
	mc.mu.Unlock()
}

// SwitchToProvider makes the named provider active.
// Returns false if no provider with that name is registered.
func (mc *MultiClient) SwitchToProvider(name string) bool {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	for i, e := range mc.entries {
		if e.providerName == name {
			mc.active = i
			return true
		}
	}
	return false
}

// Providers returns the names of all registered providers in registration order.
func (mc *MultiClient) Providers() []string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	names := make([]string, len(mc.entries))
	for i, e := range mc.entries {
		names[i] = e.providerName
	}
	return names
}

// ActiveProviderName returns the name of the currently active provider.
func (mc *MultiClient) ActiveProviderName() string {
	e := mc.activeEntry()
	if e == nil {
		return ""
	}
	return e.providerName
}

// CurrentModelDisplay returns a user-facing display string for the active
// provider and model (e.g. "Copilot: gpt-4o" or just "gpt-4o" for a single
// provider setup).
func (mc *MultiClient) CurrentModelDisplay() string {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	if len(mc.entries) == 0 {
		return ""
	}
	e := mc.entries[mc.active]
	model := entryCurrentModel(e)
	if model == "" {
		return e.providerName
	}
	if len(mc.entries) == 1 {
		return model
	}
	return e.providerName + ": " + model
}

func (mc *MultiClient) activeEntry() *multiEntry {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	if len(mc.entries) == 0 {
		return nil
	}
	return mc.entries[mc.active]
}

// Modeler is an optional interface for clients that expose their current model name.
type Modeler interface {
	Model() string
}

// entryCurrentModel tries to read the current model name from the entry's StreamCompleter.
func entryCurrentModel(e *multiEntry) string {
	if m, ok := e.sc.(Modeler); ok {
		return m.Model()
	}
	return ""
}

// Compile-time interface assertions.
var _ StreamCompleter = (*MultiClient)(nil)
var _ Completer = (*MultiClient)(nil)
var _ ModelManager = (*MultiClient)(nil)

// CompleteStream delegates to the active provider's StreamCompleter.
func (mc *MultiClient) CompleteStream(ctx context.Context, messages []Message, tools []ToolDefinition) <-chan StreamChunk {
	e := mc.activeEntry()
	if e == nil {
		ch := make(chan StreamChunk, 1)
		ch <- StreamChunk{Err: fmt.Errorf("no AI provider configured")}
		close(ch)
		return ch
	}
	return e.sc.CompleteStream(ctx, messages, tools)
}

// Complete delegates to the active provider's Completer, if supported.
func (mc *MultiClient) Complete(ctx context.Context, messages []Message, tools []ToolDefinition) (Message, error) {
	e := mc.activeEntry()
	if e == nil {
		return Message{}, fmt.Errorf("no AI provider configured")
	}
	if e.completer == nil {
		return Message{}, fmt.Errorf("provider %q does not support non-streaming completions", e.providerName)
	}
	return e.completer.Complete(ctx, messages, tools)
}

// SetModel switches the active provider (when item.Provider is set) and
// updates the model on the resulting active provider's ModelManager.
func (mc *MultiClient) SetModel(item ModelItem) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	if item.Provider != "" {
		for i, e := range mc.entries {
			if e.providerName == item.Provider {
				mc.active = i
				break
			}
		}
	}
	e := mc.entries[mc.active]
	if e.mm != nil {
		e.mm.SetModel(ModelItem{Model: item.Model})
	}
}

// ListModels returns all models across all providers, with the active
// provider's models first. Per-provider errors are silently skipped so
// that a broken provider does not prevent using the others.
func (mc *MultiClient) ListModels(ctx context.Context) ([]ModelItem, error) {
	mc.mu.RLock()
	entries := make([]*multiEntry, len(mc.entries))
	copy(entries, mc.entries)
	active := mc.active
	mc.mu.RUnlock()

	type result struct {
		items []ModelItem
	}
	results := make([]result, len(entries))

	var wg sync.WaitGroup
	for i, e := range entries {
		wg.Add(1)
		go func(i int, e *multiEntry) {
			defer wg.Done()
			if e.mm == nil {
				return
			}
			items, err := e.mm.ListModels(ctx)
			if err != nil {
				return
			}
			out := make([]ModelItem, len(items))
			for j, it := range items {
				out[j] = ModelItem{Provider: e.providerName, Model: it.Model}
			}
			results[i] = result{items: out}
		}(i, e)
	}
	wg.Wait()

	// Output: active provider first, then the rest in registration order.
	order := make([]int, 0, len(entries))
	order = append(order, active)
	for i := range entries {
		if i != active {
			order = append(order, i)
		}
	}

	var out []ModelItem
	for _, i := range order {
		out = append(out, results[i].items...)
	}
	return out, nil
}

// GetModelInfo delegates to the active provider's ModelInfoGetter if supported.
// Returns empty ModelInfo without error when the active provider doesn't support it.
func (mc *MultiClient) GetModelInfo(ctx context.Context) (ModelInfo, error) {
	e := mc.activeEntry()
	if e == nil {
		return ModelInfo{}, nil
	}
	if g, ok := e.sc.(ModelInfoGetter); ok {
		return g.GetModelInfo(ctx)
	}
	return ModelInfo{}, nil
}
