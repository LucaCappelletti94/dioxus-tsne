import { startWorkers } from './snippets/wasm-bindgen-rayon-38edf6e439f6d70d/src/workerHelpers.no-bundler.js';


/**
 * @param {number} num_threads
 * @returns {Promise<any>}
 */
export function initThreadPool(num_threads) {
    const ret = wasm.initThreadPool(num_threads);
    return ret;
}

/**
 * Registers the gloo worker so it starts handling compute messages.
 *
 * The loader calls this only after `initThreadPool` has built the rayon pool.
 * Registering earlier would let a queued compute message touch rayon first,
 * which lazily initializes the global pool to a single thread, after which
 * wasm-bindgen-rayon's real `build_global` fails and the worker is stuck
 * single threaded.
 */
export function register_worker() {
    wasm.register_worker();
}

export class wbg_rayon_PoolBuilder {
    static __wrap(ptr) {
        const obj = Object.create(wbg_rayon_PoolBuilder.prototype);
        obj.__wbg_ptr = ptr;
        wbg_rayon_PoolBuilderFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        wbg_rayon_PoolBuilderFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 0);
    }
    build() {
        wasm.wbg_rayon_poolbuilder_build(this.__wbg_ptr);
    }
    /**
     * @returns {string}
     */
    mainJS() {
        const ret = wasm.wbg_rayon_poolbuilder_mainJS(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    numThreads() {
        const ret = wasm.wbg_rayon_poolbuilder_numThreads(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    receiver() {
        const ret = wasm.wbg_rayon_poolbuilder_receiver(this.__wbg_ptr);
        return ret >>> 0;
    }
}
if (Symbol.dispose) wbg_rayon_PoolBuilder.prototype[Symbol.dispose] = wbg_rayon_PoolBuilder.prototype.free;

/**
 * @param {number} receiver
 */
export function wbg_rayon_start_worker(receiver) {
    wasm.wbg_rayon_start_worker(receiver);
}
function __wbg_get_imports(memory) {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_function_5cd60d5cf78b4eef: function(arg0) {
            const ret = typeof(arg0) === 'function';
            return ret;
        },
        __wbg___wbindgen_is_undefined_35bb9f4c7fd651d5: function(arg0) {
            const ret = arg0 === undefined;
            return ret;
        },
        __wbg___wbindgen_memory_9544558992fc5400: function() {
            const ret = wasm.memory;
            return ret;
        },
        __wbg___wbindgen_module_598c7f098f85bbd9: function() {
            const ret = wasmModule;
            return ret;
        },
        __wbg___wbindgen_rethrow_2b7cc655458909c2: function(arg0) {
            throw arg0;
        },
        __wbg___wbindgen_throw_9c31b086c2b26051: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg__wbg_cb_unref_3fa391f3fcdb55f8: function(arg0) {
            arg0._wbg_cb_unref();
        },
        __wbg_async_c2781a14e36ac9e4: function(arg0) {
            const ret = arg0.async;
            return ret;
        },
        __wbg_buffer_09bfa2e33737b5fc: function(arg0) {
            const ret = arg0.buffer;
            return ret;
        },
        __wbg_call_dfde26266607c996: function() { return handleError(function (arg0, arg1, arg2) {
            const ret = arg0.call(arg1, arg2);
            return ret;
        }, arguments); },
        __wbg_close_d293331974ac3c1c: function(arg0) {
            arg0.close();
        },
        __wbg_data_5fc79a19e47d1531: function(arg0) {
            const ret = arg0.data;
            return ret;
        },
        __wbg_data_726486ab4e13b1e8: function(arg0) {
            const ret = arg0.data;
            return ret;
        },
        __wbg_error_a6fa202b58aa1cd3: function(arg0, arg1) {
            let deferred0_0;
            let deferred0_1;
            try {
                deferred0_0 = arg0;
                deferred0_1 = arg1;
                console.error(getStringFromWasm0(arg0, arg1));
            } finally {
                wasm.__wbindgen_free(deferred0_0, deferred0_1, 1);
            }
        },
        __wbg_getRandomValues_b2176991427f6db8: function() { return handleError(function (arg0) {
            globalThis.crypto.getRandomValues(arg0);
        }, arguments); },
        __wbg_instanceof_Window_faa5cf994f49cca7: function(arg0) {
            let result;
            try {
                result = arg0 instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_length_56fcd3e2b7e0299d: function(arg0) {
            const ret = arg0.length;
            return ret;
        },
        __wbg_new_227d7c05414eb861: function() {
            const ret = new Error();
            return ret;
        },
        __wbg_new_7a51510df9f459ff: function(arg0) {
            const ret = new Int32Array(arg0);
            return ret;
        },
        __wbg_new_d8dfd33fa007511d: function(arg0, arg1) {
            try {
                var state0 = {a: arg0, b: arg1};
                var cb0 = (arg0, arg1) => {
                    const a = state0.a;
                    state0.a = 0;
                    try {
                        return wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined_______true_(a, state0.b, arg0, arg1);
                    } finally {
                        state0.a = a;
                    }
                };
                const ret = new Promise(cb0);
                return ret;
            } finally {
                state0.a = 0;
            }
        },
        __wbg_new_from_slice_269e35316ed2d061: function(arg0, arg1) {
            const ret = new Uint8Array(getArrayU8FromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_new_with_length_99887c91eae4abab: function(arg0) {
            const ret = new Uint8Array(arg0 >>> 0);
            return ret;
        },
        __wbg_new_worker_3f3757894eaf84eb: function(arg0, arg1) {
            const ret = new Worker(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg_now_81363d44c96dd239: function() {
            const ret = Date.now();
            return ret;
        },
        __wbg_of_d77919dcf5640358: function(arg0, arg1, arg2) {
            const ret = Array.of(arg0, arg1, arg2);
            return ret;
        },
        __wbg_postMessage_5e48f2cd6e9c2baa: function() { return handleError(function (arg0, arg1) {
            arg0.postMessage(arg1);
        }, arguments); },
        __wbg_postMessage_d6c0cb4a6e9bb3c2: function() { return handleError(function (arg0, arg1) {
            arg0.postMessage(arg1);
        }, arguments); },
        __wbg_prototypesetcall_5f9bdc8d75e07276: function(arg0, arg1, arg2) {
            Uint8Array.prototype.set.call(getArrayU8FromWasm0(arg0, arg1), arg2);
        },
        __wbg_queueMicrotask_78d584b53af520f5: function(arg0) {
            const ret = arg0.queueMicrotask;
            return ret;
        },
        __wbg_queueMicrotask_b39ea83c7f01971a: function(arg0) {
            queueMicrotask(arg0);
        },
        __wbg_resolve_d17db9352f5a220e: function(arg0) {
            const ret = Promise.resolve(arg0);
            return ret;
        },
        __wbg_set_onmessage_4ffd530f8e195138: function(arg0, arg1) {
            arg0.onmessage = arg1;
        },
        __wbg_set_onmessage_c767f40c3874bffb: function(arg0, arg1) {
            arg0.onmessage = arg1;
        },
        __wbg_stack_3b0d974bbf31e44f: function(arg0, arg1) {
            const ret = arg1.stack;
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        __wbg_startWorkers_622cedd0d351664e: function(arg0, arg1, arg2) {
            const ret = startWorkers(arg0, arg1, wbg_rayon_PoolBuilder.__wrap(arg2));
            return ret;
        },
        __wbg_static_accessor_GLOBAL_THIS_02344c9b09eb08a9: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_GLOBAL_ac6d4ac874d5cd54: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_SELF_9b2406c23aeb2023: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_static_accessor_URL_151cb8815849ce83: function() {
            const ret = import.meta.url;
            return ret;
        },
        __wbg_static_accessor_WINDOW_b34d2126934e16ba: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addToExternrefTable0(ret);
        },
        __wbg_subarray_7c6a0da8f3b4a1ba: function(arg0, arg1, arg2) {
            const ret = arg0.subarray(arg1 >>> 0, arg2 >>> 0);
            return ret;
        },
        __wbg_then_7b57a40e3ee05615: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_then_837494e384b37459: function(arg0, arg1) {
            const ret = arg0.then(arg1);
            return ret;
        },
        __wbg_value_9e2825f14753cda1: function(arg0) {
            const ret = arg0.value;
            return ret;
        },
        __wbg_waitAsync_94344b8da427fee1: function(arg0, arg1, arg2) {
            const ret = Atomics.waitAsync(arg0, arg1 >>> 0, arg2);
            return ret;
        },
        __wbg_waitAsync_c887dc4b662e4cf3: function() {
            const ret = Atomics.waitAsync;
            return ret;
        },
        __wbindgen_cast_0000000000000001: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 6730, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue______true_);
            return ret;
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 6750, ret: Result(Unit), inner_ret: Some(Result(Unit)) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue__core_9788cdba3bb3d89___result__Result_____wasm_bindgen_aff15ea5baa0a177___JsError___true_);
            return ret;
        },
        __wbindgen_cast_0000000000000003: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [Externref], shim_idx: 6767, ret: Unit, inner_ret: Some(Unit) }, mutable: true }) -> Externref`.
            const ret = makeMutClosure(arg0, arg1, wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___futures__task__wait_async_polyfill__MessageEvent______true_);
            return ret;
        },
        __wbindgen_cast_0000000000000004: function(arg0, arg1) {
            // Cast intrinsic for `Closure(Closure { owned: true, function: Function { arguments: [NamedExternref("MessageEvent")], shim_idx: 68, ret: Unit, inner_ret: Some(Unit) }, mutable: false }) -> Externref`.
            const ret = makeClosure(arg0, arg1, wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___web_sys_572efd2c6ea959ea___features__gen_MessageEvent__MessageEvent______true_);
            return ret;
        },
        __wbindgen_cast_0000000000000005: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return ret;
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
        __wbindgen_link_56971320a4814863: function(arg0) {
            const val = `onmessage = function (ev) {
                let [ia, index, value] = ev.data;
                ia = new Int32Array(ia.buffer);
                let result = Atomics.wait(ia, index, value);
                postMessage(result);
            };
            `;
            const ret = typeof URL.createObjectURL === 'undefined' ? "data:application/javascript," + encodeURIComponent(val) : URL.createObjectURL(new Blob([val], { type: "text/javascript" }));
            const ptr1 = passStringToWasm0(ret, wasm.__wbindgen_malloc, wasm.__wbindgen_realloc);
            const len1 = WASM_VECTOR_LEN;
            getDataViewMemory0().setInt32(arg0 + 4 * 1, len1, true);
            getDataViewMemory0().setInt32(arg0 + 4 * 0, ptr1, true);
        },
        memory: memory || new WebAssembly.Memory({initial:23,maximum:16384,shared:true}),
    };
    return {
        __proto__: null,
        "./decompositions_worker_bg.js": import0,
    };
}

function wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue______true_(arg0, arg1, arg2);
}

function wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___futures__task__wait_async_polyfill__MessageEvent______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___futures__task__wait_async_polyfill__MessageEvent______true_(arg0, arg1, arg2);
}

function wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___web_sys_572efd2c6ea959ea___features__gen_MessageEvent__MessageEvent______true_(arg0, arg1, arg2) {
    wasm.wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___web_sys_572efd2c6ea959ea___features__gen_MessageEvent__MessageEvent______true_(arg0, arg1, arg2);
}

function wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue__core_9788cdba3bb3d89___result__Result_____wasm_bindgen_aff15ea5baa0a177___JsError___true_(arg0, arg1, arg2) {
    const ret = wasm.wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___wasm_bindgen_aff15ea5baa0a177___JsValue__core_9788cdba3bb3d89___result__Result_____wasm_bindgen_aff15ea5baa0a177___JsError___true_(arg0, arg1, arg2);
    if (ret[1]) {
        throw takeFromExternrefTable0(ret[0]);
    }
}

function wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined_______true_(arg0, arg1, arg2, arg3) {
    wasm.wasm_bindgen_aff15ea5baa0a177___convert__closures_____invoke___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined___js_sys_5d98671b6e0fb8ab___Function_fn_wasm_bindgen_aff15ea5baa0a177___JsValue_____wasm_bindgen_aff15ea5baa0a177___sys__Undefined_______true_(arg0, arg1, arg2, arg3);
}

const wbg_rayon_PoolBuilderFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_wbg_rayon_poolbuilder_free(ptr, 1));

function addToExternrefTable0(obj) {
    const idx = wasm.__externref_table_alloc();
    wasm.__wbindgen_externrefs.set(idx, obj);
    return idx;
}

const CLOSURE_DTORS = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(state => wasm.__wbindgen_destroy_closure(state.a, state.b));

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer !== wasm.memory.buffer) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.buffer !== wasm.memory.buffer) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        const idx = addToExternrefTable0(e);
        wasm.__wbindgen_exn_store(idx);
    }
}

function isLikeNone(x) {
    return x === undefined || x === null;
}

function makeClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        try {
            return f(state.a, state.b, ...args);
        } finally {
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function makeMutClosure(arg0, arg1, f) {
    const state = { a: arg0, b: arg1, cnt: 1 };
    const real = (...args) => {

        // First up with a closure we increment the internal reference
        // count. This ensures that the Rust closure environment won't
        // be deallocated while we're invoking it.
        state.cnt++;
        const a = state.a;
        state.a = 0;
        try {
            return f(a, state.b, ...args);
        } finally {
            state.a = a;
            real._wbg_cb_unref();
        }
    };
    real._wbg_cb_unref = () => {
        if (--state.cnt === 0) {
            wasm.__wbindgen_destroy_closure(state.a, state.b);
            state.a = 0;
            CLOSURE_DTORS.unregister(state);
        }
    };
    CLOSURE_DTORS.register(real, state, state);
    return real;
}

function passStringToWasm0(arg, malloc, realloc) {
    if (realloc === undefined) {
        const buf = cachedTextEncoder.encode(arg);
        const ptr = malloc(buf.length, 1) >>> 0;
        getUint8ArrayMemory0().subarray(ptr, ptr + buf.length).set(buf);
        WASM_VECTOR_LEN = buf.length;
        return ptr;
    }

    let len = arg.length;
    let ptr = malloc(len, 1) >>> 0;

    const mem = getUint8ArrayMemory0();

    let offset = 0;

    for (; offset < len; offset++) {
        const code = arg.charCodeAt(offset);
        if (code > 0x7F) break;
        mem[ptr + offset] = code;
    }
    if (offset !== len) {
        if (offset !== 0) {
            arg = arg.slice(offset);
        }
        ptr = realloc(ptr, len, len = offset + arg.length * 3, 1) >>> 0;
        const view = getUint8ArrayMemory0().subarray(ptr + offset, ptr + len);
        const ret = cachedTextEncoder.encodeInto(arg, view);

        offset += ret.written;
        ptr = realloc(ptr, len, offset, 1) >>> 0;
    }

    WASM_VECTOR_LEN = offset;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = (typeof TextDecoder !== 'undefined' ? new TextDecoder('utf-8', { ignoreBOM: true, fatal: true }) : undefined);
if (cachedTextDecoder) cachedTextDecoder.decode();

const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().slice(ptr, ptr + len));
}

const cachedTextEncoder = (typeof TextEncoder !== 'undefined' ? new TextEncoder() : undefined);

if (cachedTextEncoder) {
    cachedTextEncoder.encodeInto = function (arg, view) {
        const buf = cachedTextEncoder.encode(arg);
        view.set(buf);
        return {
            read: arg.length,
            written: buf.length
        };
    };
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module, thread_stack_size) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    if (typeof thread_stack_size !== 'undefined' && (typeof thread_stack_size !== 'number' || thread_stack_size === 0 || thread_stack_size % 65536 !== 0)) {
        throw new Error('invalid stack size');
    }

    wasm.__wbindgen_start(thread_stack_size);
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module, memory, thread_stack_size} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports(memory);
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module, thread_stack_size);
}

async function __wbg_init(module_or_path, memory) {
    if (wasm !== undefined) return wasm;

    let thread_stack_size
    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path, memory, thread_stack_size} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }


    const imports = __wbg_get_imports(memory);

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module, thread_stack_size);
}

export { initSync, __wbg_init as default };
