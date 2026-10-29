//! `exact_bm25`: SQLite's FTS5 `bm25()` over statistics the index keeps
//! itself.
//!
//! The contentless `fts` table (`contentless_delete = 1`) never takes a
//! deleted row out of its averages record, so the row count and token totals
//! the built-in `bm25()` reads drift as chunks are replaced. `exact_bm25`
//! computes the built-in's formula — `fts5Bm25Function` and
//! `fts5Bm25GetData` in the bundled `sqlite3.c` — with every input taken
//! where the built-in takes it except those two, which arrive as arguments
//! from the `bm25_stats` row:
//!
//! ```sql
//! exact_bm25(fts, rows, column_0_tokens, ..., column_n_tokens [, column weights...])
//! ```
//!
//! The trailing column weights are the built-in's own arguments, shifted
//! past the statistics; a column without one weighs 1.

use std::ffi::{CStr, c_int, c_void};
use std::{ptr, slice};

use rusqlite::Connection;
use rusqlite::ffi::{
    self, Fts5Context, Fts5ExtensionApi, fts5_api, sqlite3_context, sqlite3_value,
};

use super::Error;

/// The function's SQL name.
const NAME: &CStr = c"exact_bm25";

/// BM25's `k1`, as `fts5Bm25Function` sets it.
const K1: f64 = 1.2;

/// BM25's `b`, as `fts5Bm25Function` sets it.
const B: f64 = 0.75;

/// The least IDF a phrase gets, as `fts5Bm25GetData` floors it: a phrase in
/// more than half the rows would otherwise score negative.
const IDF_FLOOR: f64 = 1e-6;

/// Registers `exact_bm25` on `conn` through FTS5's extension API.
///
/// # Errors
///
/// When SQLite cannot hand out its `fts5_api` or rejects the registration.
pub(super) fn register(conn: &Connection) -> Result<(), Error> {
    // SAFETY: the handle is `conn`'s open database; it is used only inside
    // this call, while `conn` is borrowed and therefore open.
    let db = unsafe { conn.handle() };
    // SAFETY: `db` is an open database handle (above).
    let api = unsafe { fts5_api_of(db) }?;
    // SAFETY: `api` is the connection's `fts5_api`, valid while the
    // connection is open; every `fts5_api` version fills `xCreateFunction`.
    // `exact_bm25` has the `fts5_extension_function` signature, takes no user
    // data and so needs no destructor.
    let rc = unsafe {
        let create = (*api).xCreateFunction.unwrap_unchecked();
        create(api, NAME.as_ptr(), ptr::null_mut(), Some(exact_bm25), None)
    };
    if rc == ffi::SQLITE_OK {
        Ok(())
    } else {
        // SAFETY: `db` is open (above).
        Err(unsafe { failure(db, rc) })
    }
}

/// The connection's `fts5_api`, which FTS5 writes through the pointer bound
/// to `SELECT fts5(?1)` under the type `fts5_api_ptr`.
///
/// # Safety
///
/// `db` is an open database handle.
unsafe fn fts5_api_of(db: *mut ffi::sqlite3) -> Result<*mut fts5_api, Error> {
    let mut api: *mut fts5_api = ptr::null_mut();
    let mut stmt = ptr::null_mut();
    // SAFETY: `db` is open (the caller's contract); the SQL is a
    // nul-terminated literal and `stmt` receives the prepared statement.
    let rc = unsafe {
        ffi::sqlite3_prepare_v2(
            db,
            c"SELECT fts5(?1)".as_ptr(),
            -1,
            &raw mut stmt,
            ptr::null_mut(),
        )
    };
    if rc != ffi::SQLITE_OK {
        // SAFETY: `db` is open.
        return Err(unsafe { failure(db, rc) });
    }
    // SAFETY: `stmt` was prepared above and has one parameter. The bound
    // pointer is `api`, which outlives the statement: it is finalized before
    // this function returns. The type name is a static literal, as
    // `sqlite3_bind_pointer` requires.
    let rc = unsafe {
        ffi::sqlite3_bind_pointer(
            stmt,
            1,
            (&raw mut api).cast(),
            c"fts5_api_ptr".as_ptr(),
            None,
        );
        ffi::sqlite3_step(stmt);
        ffi::sqlite3_finalize(stmt)
    };
    if rc != ffi::SQLITE_OK {
        // SAFETY: `db` is open.
        return Err(unsafe { failure(db, rc) });
    }
    // A successful step has written the connection's `fts5_api` through the
    // bound pointer (`fts5Fts5Func`).
    Ok(api)
}

/// The error SQLite reported on `db` for `rc`.
///
/// # Safety
///
/// `db` is an open database handle.
unsafe fn failure(db: *mut ffi::sqlite3, rc: c_int) -> Error {
    // SAFETY: `db` is open, so `sqlite3_errmsg` returns a nul-terminated
    // string valid until the next call on `db`; it is copied at once.
    let message = unsafe { CStr::from_ptr(ffi::sqlite3_errmsg(db)) }
        .to_string_lossy()
        .into_owned();
    Error::Sqlite(rusqlite::Error::SqliteFailure(
        ffi::Error::new(rc),
        Some(message),
    ))
}

/// What `fts5Bm25GetData` computes once per query: the average row length
/// and each phrase's IDF, plus the per-row phrase-frequency scratch array.
struct QueryData {
    avgdl: f64,
    idf: Vec<f64>,
    freq: Vec<f64>,
}

/// Why a row has no score.
enum Failure {
    /// An FTS5 API call failed with this code.
    Code(c_int),
    /// The SQL call is malformed.
    Usage(&'static CStr),
}

/// Turns an FTS5 API return code into a [`Failure`].
fn ok(rc: c_int) -> Result<(), Failure> {
    if rc == ffi::SQLITE_OK {
        Ok(())
    } else {
        Err(Failure::Code(rc))
    }
}

/// The FTS5 auxiliary function: `fts5Bm25Function` with the row count and
/// token totals taken from its arguments.
///
/// # Safety
///
/// FTS5 calls it, during a query's row step, with its extension API table,
/// the cursor's context, the result context and `n_val` trailing arguments
/// at `ap_val`.
unsafe extern "C" fn exact_bm25(
    api: *const Fts5ExtensionApi,
    fts: *mut Fts5Context,
    ctx: *mut sqlite3_context,
    n_val: c_int,
    ap_val: *mut *mut sqlite3_value,
) {
    // SAFETY: FTS5 passes its own API table, valid for the call.
    let api = unsafe { &*api };
    // SAFETY: `ap_val` points at `n_val` (non-negative) argument values:
    // FTS5 passes the SQL call's arguments after the table's.
    let args = unsafe { slice::from_raw_parts(ap_val, n_val as usize) };
    // SAFETY: `fts` is the cursor context of this call and `args` its values.
    let outcome = unsafe { row_score(api, fts, args) };
    // SAFETY: `ctx` is this call's result context.
    unsafe {
        match outcome {
            Ok(score) => ffi::sqlite3_result_double(ctx, -score),
            Err(Failure::Code(rc)) => ffi::sqlite3_result_error_code(ctx, rc),
            Err(Failure::Usage(message)) => ffi::sqlite3_result_error(ctx, message.as_ptr(), -1),
        }
    }
}

/// The current row's BM25 sum, before `fts5Bm25Function` negates it.
///
/// # Safety
///
/// `api` and `fts` are the ones FTS5 passed to this call of [`exact_bm25`],
/// and `args` its trailing arguments.
unsafe fn row_score(
    api: &Fts5ExtensionApi,
    fts: *mut Fts5Context,
    args: &[*mut sqlite3_value],
) -> Result<f64, Failure> {
    // SAFETY: FTS5's extension API table fills every method it declares.
    let x_column_count = unsafe { api.xColumnCount.unwrap_unchecked() };
    // SAFETY: as above.
    let x_inst_count = unsafe { api.xInstCount.unwrap_unchecked() };
    // SAFETY: as above.
    let x_inst = unsafe { api.xInst.unwrap_unchecked() };
    // SAFETY: as above.
    let x_column_size = unsafe { api.xColumnSize.unwrap_unchecked() };

    // SAFETY: `fts` is this call's cursor context; a column count is
    // non-negative.
    let n_col = unsafe { x_column_count(fts) } as usize;
    let Some((stats, weights)) = args.split_at_checked(1 + n_col) else {
        return Err(Failure::Usage(
            c"exact_bm25 takes the row count and each column's token total",
        ));
    };
    // SAFETY: as this function's contract.
    let data = unsafe { query_data(api, fts, stats) }?;

    // fts5Bm25Function: the phrase frequency of each phrase in this row, an
    // instance in column `ic` counting that column's weight.
    data.freq.fill(0.0);
    let mut n_inst: c_int = 0;
    // SAFETY: `fts` is this call's cursor context; `n_inst` is a valid out
    // pointer.
    ok(unsafe { x_inst_count(fts, &raw mut n_inst) })?;
    for i in 0..n_inst {
        let (mut ip, mut ic, mut io): (c_int, c_int, c_int) = (0, 0, 0);
        // SAFETY: `i` is below the instance count FTS5 reported; the out
        // pointers are valid.
        ok(unsafe { x_inst(fts, i, &raw mut ip, &raw mut ic, &raw mut io) })?;
        // An instance's phrase and column are non-negative indexes below the
        // phrase and column counts.
        let w = match weights.get(ic as usize) {
            // SAFETY: the value is one of this call's arguments.
            Some(&value) => unsafe { ffi::sqlite3_value_double(value) },
            None => 1.0,
        };
        data.freq[ip as usize] += w;
    }

    // fts5Bm25Function: the row's total size in tokens, over every column.
    let mut n_tok: c_int = 0;
    // SAFETY: `fts` is this call's cursor context; `n_tok` is a valid out
    // pointer.
    ok(unsafe { x_column_size(fts, -1, &raw mut n_tok) })?;
    let d = f64::from(n_tok);

    let mut score = 0.0;
    for (idf, freq) in data.idf.iter().zip(&data.freq) {
        score += idf * ((freq * (K1 + 1.0)) / (freq + K1 * (1.0 - B + B * d / data.avgdl)));
    }
    Ok(score)
}

/// This query's [`QueryData`], computed on its first row and kept as the
/// cursor's auxiliary data, as `fts5Bm25GetData` keeps its own.
///
/// # Safety
///
/// As [`row_score`]; `stats` holds the row count then one token total per
/// column.
unsafe fn query_data<'a>(
    api: &Fts5ExtensionApi,
    fts: *mut Fts5Context,
    stats: &[*mut sqlite3_value],
) -> Result<&'a mut QueryData, Failure> {
    // SAFETY: FTS5's extension API table fills every method it declares.
    let x_get_auxdata = unsafe { api.xGetAuxdata.unwrap_unchecked() };
    // SAFETY: as above.
    let x_set_auxdata = unsafe { api.xSetAuxdata.unwrap_unchecked() };
    // SAFETY: as above.
    let x_phrase_count = unsafe { api.xPhraseCount.unwrap_unchecked() };
    // SAFETY: as above.
    let x_query_phrase = unsafe { api.xQueryPhrase.unwrap_unchecked() };

    // SAFETY: `fts` is this call's cursor context. The only auxiliary data
    // `exact_bm25` ever stores on a cursor is a boxed `QueryData`, which
    // lives until FTS5 runs `drop_query_data` on it at the end of the query;
    // the returned borrow does not outlive this row's call.
    let stored = unsafe { x_get_auxdata(fts, 0) }.cast::<QueryData>();
    if !stored.is_null() {
        // SAFETY: as above; nothing else borrows it during this call.
        return Ok(unsafe { &mut *stored });
    }

    // SAFETY: the values are this call's arguments.
    let n_row = unsafe { ffi::sqlite3_value_int64(stats[0]) };
    if n_row < 1 {
        // fts5Bm25GetData asserts a matched table has a row.
        return Err(Failure::Usage(c"exact_bm25 was given a row count below 1"));
    }
    let n_token: i64 = stats[1..]
        .iter()
        // SAFETY: the values are this call's arguments.
        .map(|&value| unsafe { ffi::sqlite3_value_int64(value) })
        .sum();
    // SAFETY: `fts` is this call's cursor context; a phrase count is
    // non-negative.
    let n_phrase = unsafe { x_phrase_count(fts) } as usize;

    let mut idf = Vec::with_capacity(n_phrase);
    for i in 0..n_phrase {
        let mut n_hit: i64 = 0;
        let phrase = i as c_int; // below a c_int count
        // SAFETY: `phrase` is below the phrase count; `count_row` only
        // increments the `i64` the user data points at, which outlives the
        // call.
        ok(unsafe { x_query_phrase(fts, phrase, (&raw mut n_hit).cast(), Some(count_row)) })?;
        // fts5Bm25GetData: IDF = log((N - nHit + 0.5) / (nHit + 0.5)),
        // floored, with `N - nHit` an integer subtraction.
        let value = (((n_row - n_hit) as f64 + 0.5) / (n_hit as f64 + 0.5)).ln();
        idf.push(if value <= 0.0 { IDF_FLOOR } else { value });
    }
    let data = Box::into_raw(Box::new(QueryData {
        avgdl: n_token as f64 / n_row as f64,
        idf,
        freq: vec![0.0; n_phrase],
    }));
    // SAFETY: `data` is a fresh box handed to FTS5 with its destructor; on
    // failure FTS5 runs the destructor itself.
    ok(unsafe { x_set_auxdata(fts, data.cast(), Some(drop_query_data)) })?;
    // SAFETY: FTS5 now holds `data` until the query ends; this call's borrow
    // ends before then.
    Ok(unsafe { &mut *data })
}

/// `xQueryPhrase` callback: counts one matching row, as `fts5CountCb`.
///
/// # Safety
///
/// `user` points at the `i64` counter [`query_data`] passed.
unsafe extern "C" fn count_row(
    _api: *const Fts5ExtensionApi,
    _fts: *mut Fts5Context,
    user: *mut c_void,
) -> c_int {
    // SAFETY: as this function's contract.
    unsafe { *user.cast::<i64>() += 1 };
    ffi::SQLITE_OK
}

/// Destructor FTS5 runs on a cursor's [`QueryData`].
///
/// # Safety
///
/// `data` came from `Box::into_raw` in [`query_data`] and is dropped once.
unsafe extern "C" fn drop_query_data(data: *mut c_void) {
    // SAFETY: as this function's contract.
    drop(unsafe { Box::from_raw(data.cast::<QueryData>()) });
}
