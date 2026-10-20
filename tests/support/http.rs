//! A one-shot HTTP server and model-archive builders for tests that drive
//! `ModelStore` without the network.

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

/// A tar.zst archive whose only entry is a symlink.
pub fn symlink_archive() -> Vec<u8> {
    let encoder = zstd::stream::write::Encoder::new(Vec::new(), 0).unwrap();
    let mut builder = tar::Builder::new(encoder);
    let mut header = tar::Header::new_gnu();
    header.set_entry_type(tar::EntryType::Symlink);
    header.set_size(0);
    header.set_mode(0o777);
    builder
        .append_link(&mut header, "evil", "/etc/passwd")
        .unwrap();
    builder.into_inner().unwrap().finish().unwrap()
}

/// One-shot HTTP/1.1 server: replies `status` and declares
/// `content_length`, then writes `body` and closes the connection.
pub fn serve_once(
    status: &str,
    content_length: u64,
    body: Vec<u8>,
) -> (String, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/model.tar.zst", listener.local_addr().unwrap());
    let status = status.to_string();
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
        let head = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {content_length}\r\nConnection: close\r\n\r\n"
        );
        if stream.write_all(head.as_bytes()).is_err() {
            return;
        }
        let _ = stream.write_all(&body);
    });
    (url, handle)
}
