use aideen_core::protocol::{AckKind, KeyDelegation, NetMsg, SignedUpdate};

#[test]
fn test_netmsg_encode_decode_hello() {
    let msg = NetMsg::Hello {
        node_id: [1u8; 32],
        protocol: 1,
        bundle_version: 0,
        bundle_hash: [2u8; 32],
    };

    let bytes = msg.encode().expect("Debe poder codificar");
    let decoded = NetMsg::decode(&bytes).expect("Debe poder decodificar");

    match decoded {
        NetMsg::Hello {
            node_id,
            protocol,
            bundle_version,
            bundle_hash,
        } => {
            assert_eq!(node_id, [1u8; 32]);
            assert_eq!(protocol, 1);
            assert_eq!(bundle_version, 0);
            assert_eq!(bundle_hash, [2u8; 32]);
        }
        _ => panic!("Decoded wrong enum variant"),
    }
}

#[test]
fn test_netmsg_encode_decode_delegation() {
    let del = KeyDelegation {
        epoch: 42,
        critic_pk: [3u8; 32],
        valid_from_unix: 1000,
        valid_to_unix: 2000,
        signature_by_root: vec![0, 1, 2, 3],
    };

    let msg = NetMsg::Delegation(del);

    let bytes = msg.encode().unwrap();
    let decoded = NetMsg::decode(&bytes).unwrap();

    match decoded {
        NetMsg::Delegation(d) => {
            assert_eq!(d.epoch, 42);
            assert_eq!(d.critic_pk, [3u8; 32]);
            assert_eq!(d.signature_by_root, vec![0, 1, 2, 3]);
        }
        _ => panic!("Decoded wrong enum variant"),
    }
}

#[test]
fn test_netmsg_encode_decode_update() {
    let upd = SignedUpdate {
        version: 1,
        target_id: "test".to_string(),
        bundle_version: 1,
        bundle_hash: [4u8; 32],
        base_model_hash: [5u8; 32],
        prev_update_hash: [6u8; 32],
        payload: vec![10, 20, 30],
        signature: vec![40, 50, 60],
    };

    let msg = NetMsg::Update(upd);

    let bytes = msg.encode().unwrap();
    let decoded = NetMsg::decode(&bytes).unwrap();

    match decoded {
        NetMsg::Update(u) => {
            assert_eq!(u.version, 1);
            assert_eq!(u.target_id, "test");
            assert_eq!(u.payload, vec![10, 20, 30]);
        }
        _ => panic!("Decoded wrong enum variant"),
    }
}

#[test]
fn test_netmsg_encode_decode_ack() {
    let msg = NetMsg::Ack {
        kind: AckKind::Update,
        version: 99,
        ok: true,
    };

    let bytes = msg.encode().unwrap();
    let decoded = NetMsg::decode(&bytes).unwrap();

    match decoded {
        NetMsg::Ack { kind, version, ok } => {
            assert_eq!(kind, AckKind::Update);
            assert_eq!(version, 99);
            assert!(ok);
        }
        _ => panic!("Decoded wrong enum variant"),
    }
}
