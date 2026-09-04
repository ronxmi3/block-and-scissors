module haircut_escrow::haircut_escrow;

use sui::balance::Balance;
use sui::coin::{Self, Coin};
use sui::sui::SUI;


// ---------------------------------------------------------
// ERROR CODES
// ---------------------------------------------------------

const E_ZERO_PAYMENT: u64 = 0;
const E_INVALID_THRESHOLD: u64 = 1;
const E_ALREADY_RESOLVED: u64 = 2;
const E_INVALID_SCORE: u64 = 3;


// ---------------------------------------------------------
// STATUS CODES
// ---------------------------------------------------------

const STATUS_PENDING: u8 = 0;
const STATUS_BARBER_PAID: u8 = 1;
const STATUS_CUSTOMER_REFUNDED: u8 = 2;


// ---------------------------------------------------------
// ORACLE CAPABILITY
// ---------------------------------------------------------

// Whoever owns this object is allowed to resolve escrows.
public struct OracleCap has key, store {
    id: UID,
}


// ---------------------------------------------------------
// ESCROW OBJECT
// ---------------------------------------------------------

public struct HaircutEscrow has key {
    id: UID,

    customer: address,
    barber: address,

    // Actual SUI locked inside the escrow.
    funds: Balance<SUI>,

    // Minimum score required for barber payout.
    threshold: u8,

    // Score submitted by our oracle/backend.
    score: u8,

    // 0 = pending
    // 1 = barber paid
    // 2 = customer refunded
    status: u8,
}


// ---------------------------------------------------------
// INITIALIZATION
// ---------------------------------------------------------

// Runs once when the package is first published.
//
// Creates the OracleCap and gives it to the publisher.
fun init(ctx: &mut TxContext) {
    let oracle_cap = OracleCap {
        id: object::new(ctx),
    };

    transfer::public_transfer(
        oracle_cap,
        ctx.sender(),
    );
}


// ---------------------------------------------------------
// CREATE ESCROW
// ---------------------------------------------------------

public fun create_escrow(
    barber: address,
    payment: Coin<SUI>,
    threshold: u8,
    ctx: &mut TxContext,
) {
    // Customer must actually send some SUI.
    assert!(
        payment.value() > 0,
        E_ZERO_PAYMENT,
    );

    // Threshold must be between 1 and 100.
    assert!(
        threshold > 0 && threshold <= 100,
        E_INVALID_THRESHOLD,
    );

    // Create the escrow object.
    let escrow = HaircutEscrow {
        id: object::new(ctx),

        // Whoever calls this function is the customer.
        customer: ctx.sender(),

        barber,

        // Convert Coin<SUI> into Balance<SUI>
        // so it can be stored inside the escrow object.
        funds: payment.into_balance(),

        threshold,

        // No score has been submitted yet.
        score: 0,

        status: STATUS_PENDING,
    };

    // Share the escrow so the oracle can access it later.
    transfer::share_object(escrow);
}


// ---------------------------------------------------------
// RESOLVE ESCROW
// ---------------------------------------------------------

public fun resolve_escrow(
    escrow: &mut HaircutEscrow,

    // Requiring this proves the caller has oracle authority.
    _oracle_cap: &OracleCap,

    score: u8,

    ctx: &mut TxContext,
) {
    // Score must be between 0 and 100.
    assert!(
        score <= 100,
        E_INVALID_SCORE,
    );

    // Prevent resolving the same escrow twice.
    assert!(
        escrow.status == STATUS_PENDING,
        E_ALREADY_RESOLVED,
    );

    // Store the score on-chain.
    escrow.score = score;

    // Remove all SUI from the internal escrow balance.
    let payout_balance = escrow.funds.withdraw_all();

    // Turn the Balance<SUI> back into a transferable Coin<SUI>.
    let payout = coin::from_balance(
        payout_balance,
        ctx,
    );

    // Decide where the money goes.
    if (score >= escrow.threshold) {
        escrow.status = STATUS_BARBER_PAID;

        // Barber gets paid.
        transfer::public_transfer(
            payout,
            escrow.barber,
        );
    } else {
        escrow.status = STATUS_CUSTOMER_REFUNDED;

        // Customer gets refunded.
        transfer::public_transfer(
            payout,
            escrow.customer,
        );
    };
}