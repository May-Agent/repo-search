//! Scripted HTTP servers and a model-archive builder for the daemon tests
//! that download the model without the network.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::thread;

/// A tar.zst archive in memory with `entries` as `(path, contents)` files.
pub fn archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let encoder = zstd::stream::write::Encoder::new(Vec::new(), 0).unwrap();
    let mut builder = tar::Builder::new(encoder);
    for (name, bytes) in entries {
        let mut header = tar::Header::new_gnu();
        header.set_size(bytes.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, name, *bytes).unwrap();
    }
    builder.into_inner().unwrap().finish().unwrap()
}

/// A server that writes the head and half the body, then holds the
/// connection open until `release` fires — a download visibly mid-flight —
/// then writes the rest and closes.
#[allow(dead_code)] // used by the socket-level daemon tests only
pub fn serve_gated(
    content_length: u64,
    body: Vec<u8>,
    release: std::sync::mpsc::Receiver<()>,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.tar.zst", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = Vec::new();
        let mut buf = [0u8; 1024];
        while !request.windows(4).any(|w| w == b"\r\n\r\n") {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => return,
                Ok(n) => request.extend_from_slice(&buf[..n]),
            }
        }
        let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {content_length}\r\n\r\n");
        if stream.write_all(head.as_bytes()).is_err() {
            return;
        }
        let split = body.len() / 2;
        if stream.write_all(&body[..split]).is_err() {
            return;
        }
        let _ = release.recv();
        let _ = stream.write_all(&body[split..]);
    });
    (url, handle)
}

/// A multi-response HTTP/1.1 server: each accepted connection gets the next
/// `(status, content_length, body)` in `responses`; later connections are
/// refused by closing. Used to script download attempts.
#[allow(dead_code)] // used by the socket-level daemon tests
pub fn serve_sequence(
    responses: Vec<(&'static str, u64, Vec<u8>)>,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.tar.zst", listener.local_addr().unwrap());
    let handle = thread::spawn(move || {
        for (status, content_length, body) in responses {
            serve(&listener, status, content_length, &body);
        }
    });
    (url, handle)
}

/// Serves one connection: reads the request, replies `status` and
/// `content_length`, writes `body` and closes.
fn serve(listener: &TcpListener, status: &str, content_length: u64, body: &[u8]) {
    let (mut stream, _) = listener.accept().unwrap();
    let mut request = Vec::new();
    let mut buf = [0u8; 1024];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut buf) {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&buf[..n]),
        }
    }
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let _ = stream.write_all(body);
}
