//! The tests in this file run the standard set of test vectors from upstream:
//! https://github.com/BLAKE2/BLAKE2/blob/320c325437539ae91091ce62efec1913cd8093c2/testvectors/blake2-kat.json
//!
//! Currently those cover default hashing and keyed hashing in BLAKE2b and BLAKE2sp. But they don't
//! test the other associated data features, and they don't test any inputs longer than a couple
//! blocks.

extern crate blake2s_simd;
extern crate hex;
#[macro_use]
extern crate lazy_static;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

lazy_static! {
    static ref TEST_CASES: Vec<TestCase> =
        serde_json::from_str(include_str!("blake2-kat.json")).unwrap();
}

#[derive(Debug, Serialize, Deserialize)]
struct TestCase {
    hash: String,
    #[serde(rename = "in")]
    in_: String,
    key: String,
    out: String,
}

#[test]
fn blake2s_vectors() {
    let mut count = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2s" {
            println!("case {}, input {:?}, key {:?}", count, case.in_, case.key);
            let input_bytes = hex::decode(&case.in_).unwrap();
            let output = if case.key.is_empty() {
                blake2s_simd::blake2s(&input_bytes)
            } else {
                let key_bytes = hex::decode(&case.key).unwrap();
                blake2s_simd::Params::new()
                    .key(&key_bytes)
                    .to_state()
                    .update(&input_bytes)
                    .finalize()
            };
            assert_eq!(case.out, &*output.to_hex());
            count += 1;
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the number of test vectors
    // changes in the future, we'll need to update this count.
    assert_eq!(512, count);
}

#[test]
fn blake2sp_vectors() {
    let mut count = 0u64;
    for case in TEST_CASES.iter() {
        if &case.hash == "blake2sp" {
            println!("case {}, input {:?}, key {:?}", count, case.in_, case.key);
            let input_bytes = hex::decode(&case.in_).unwrap();
            let output = if case.key.is_empty() {
                blake2s_simd::blake2sp::blake2sp(&input_bytes)
            } else {
                let key_bytes = hex::decode(&case.key).unwrap();
                blake2s_simd::blake2sp::Params::new()
                    .key(&key_bytes)
                    .to_state()
                    .update(&input_bytes)
                    .finalize()
            };
            assert_eq!(case.out, &*output.to_hex());
            count += 1;
        }
    }

    // Make sure we don't accidentally skip all the tests somehow. If the number of test vectors
    // changes in the future, we'll need to update this count.
    assert_eq!(512, count);
}
