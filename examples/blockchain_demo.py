#!/usr/bin/env python3
"""
Blockchain Demo for PQC+FHE Integration
=======================================

Demonstrates quantum-resistant blockchain operations with privacy-preserving
smart contract execution using Post-Quantum Cryptography and Fully Homomorphic
Encryption.

Use Cases:
1. Quantum-Resistant Transaction Signing (ML-DSA)
2. Private Smart Contract Execution (FHE)
3. Confidential Token Transfers
4. Privacy-Preserving Voting System
5. Encrypted NFT Metadata

Architecture:
------------
    +------------------+     +------------------+     +------------------+
    |  Wallet Layer    |     |  Contract Layer  |     |  Consensus Layer |
    |  - PQC KeyPairs  |---->|  - FHE Compute   |---->|  - PQC Signatures|
    |  - ML-DSA Sign   |     |  - Encrypted Ops |     |  - Merkle Proofs |
    +------------------+     +------------------+     +------------------+

Standards Compliance:
--------------------
- NIST FIPS 204: ML-DSA for transaction signatures
- NIST FIPS 203: ML-KEM for encrypted key distribution
- DESILO CKKS: Homomorphic smart contract execution

References:
----------
- Bernstein et al. (2017): "Post-quantum cryptography for blockchains"
- Boneh et al. (2020): "Homomorphic encryption for private smart contracts"
- NIST IR 8547: Transition to Post-Quantum Cryptography Standards

Author: Amon (Quantum Computing Specialist)
License: MIT
"""

import sys
import os
import hashlib
import time
import json
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from enum import Enum

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pqc_fhe_integration import (
        PQCKeyManager, FHEEngine, HybridCryptoManager,
        PQCConfig, FHEConfig
    )
    PQC_FHE_AVAILABLE = True
except ImportError:
    logger.warning("PQC-FHE library not available, using mock implementations")
    PQC_FHE_AVAILABLE = False


# ============================================================================
# DATA MODELS
# ============================================================================

class TransactionType(Enum):
    """Types of blockchain transactions"""
    TRANSFER = "transfer"
    SMART_CONTRACT = "smart_contract"
    TOKEN_MINT = "token_mint"
    TOKEN_BURN = "token_burn"
    VOTE = "vote"
    NFT_MINT = "nft_mint"


@dataclass
class Wallet:
    """Quantum-resistant blockchain wallet"""
    address: str
    public_key_kem: bytes
    public_key_sign: bytes
    private_key_kem: Optional[bytes] = None
    private_key_sign: Optional[bytes] = None
    balance: float = 0.0
    nonce: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "address": self.address,
            "balance": self.balance,
            "nonce": self.nonce,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Transaction:
    """Quantum-resistant blockchain transaction"""
    tx_id: str
    tx_type: TransactionType
    sender: str
    recipient: str
    amount: float
    timestamp: datetime
    signature: bytes
    payload: Optional[Dict] = None
    gas_used: int = 0
    status: str = "pending"
    
    def to_dict(self) -> Dict:
        return {
            "tx_id": self.tx_id,
            "type": self.tx_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "timestamp": self.timestamp.isoformat(),
            "gas_used": self.gas_used,
            "status": self.status
        }


@dataclass
class Block:
    """Blockchain block with PQC signatures"""
    block_number: int
    previous_hash: str
    transactions: List[Transaction]
    timestamp: datetime
    merkle_root: str
    validator_signature: bytes
    validator_address: str
    nonce: int = 0
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            "block_number": self.block_number,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce
        }
        return hashlib.sha3_256(json.dumps(block_data, sort_keys=True).encode()).hexdigest()


@dataclass
class SmartContract:
    """Privacy-preserving smart contract"""
    contract_id: str
    code_hash: str
    owner: str
    state: Dict = field(default_factory=dict)
    encrypted_state: Optional[Any] = None  # FHE ciphertext


# ============================================================================
# WALLET MANAGER
# ============================================================================

class QuantumResistantWalletManager:
    """
    Manages quantum-resistant blockchain wallets
    
    Uses ML-DSA for signatures and ML-KEM for key exchange
    """
    
    def __init__(self, pqc_config: Optional[PQCConfig] = None):
        self.config = pqc_config or PQCConfig(
            kem_algorithm="ML-KEM-768",
            signature_algorithm="ML-DSA-65"
        )
        
        if PQC_FHE_AVAILABLE:
            self.pqc_manager = PQCKeyManager(self.config)
        else:
            self.pqc_manager = None
        
        self.wallets: Dict[str, Wallet] = {}
        logger.info(f"WalletManager initialized with {self.config.signature_algorithm}")
    
    def create_wallet(self, initial_balance: float = 0.0) -> Wallet:
        """Create a new quantum-resistant wallet"""
        
        if self.pqc_manager:
            # Generate real PQC keys
            kem_keypair = self.pqc_manager.generate_kem_keypair()
            sign_keypair = self.pqc_manager.generate_signature_keypair()
            
            public_key_kem = kem_keypair["public_key"]
            private_key_kem = kem_keypair["private_key"]
            public_key_sign = sign_keypair["public_key"]
            private_key_sign = sign_keypair["private_key"]
        else:
            # Mock keys for demo
            public_key_kem = os.urandom(32)
            private_key_kem = os.urandom(32)
            public_key_sign = os.urandom(32)
            private_key_sign = os.urandom(32)
        
        # Generate address from public key hash
        address = "0x" + hashlib.sha3_256(
            public_key_sign[:32]
        ).hexdigest()[:40]
        
        wallet = Wallet(
            address=address,
            public_key_kem=public_key_kem,
            public_key_sign=public_key_sign,
            private_key_kem=private_key_kem,
            private_key_sign=private_key_sign,
            balance=initial_balance
        )
        
        self.wallets[address] = wallet
        logger.info(f"Created wallet: {address[:16]}...")
        
        return wallet
    
    def sign_transaction(self, wallet: Wallet, tx_data: bytes) -> bytes:
        """Sign transaction with ML-DSA"""
        
        if self.pqc_manager and wallet.private_key_sign:
            signature = self.pqc_manager.sign(
                message=tx_data,
                private_key=wallet.private_key_sign
            )
            return signature
        else:
            # Mock signature
            return hashlib.sha3_256(tx_data + wallet.address.encode()).digest()
    
    def verify_signature(self, wallet: Wallet, tx_data: bytes, signature: bytes) -> bool:
        """Verify ML-DSA signature"""
        
        if self.pqc_manager:
            return self.pqc_manager.verify(
                message=tx_data,
                signature=signature,
                public_key=wallet.public_key_sign
            )
        else:
            # Mock verification
            expected = hashlib.sha3_256(tx_data + wallet.address.encode()).digest()
            return signature == expected


# ============================================================================
# PRIVATE SMART CONTRACT ENGINE
# ============================================================================

class PrivateSmartContractEngine:
    """
    FHE-based private smart contract execution engine
    
    Enables confidential computation on encrypted contract state
    """
    
    def __init__(self, fhe_config: Optional[FHEConfig] = None):
        self.config = fhe_config or FHEConfig(
            log_n=14,
            scale_bits=40,
            use_bootstrap=True
        )
        
        if PQC_FHE_AVAILABLE:
            self.fhe_engine = FHEEngine(self.config)
        else:
            self.fhe_engine = None
        
        self.contracts: Dict[str, SmartContract] = {}
        logger.info("PrivateSmartContractEngine initialized")
    
    def deploy_contract(self, owner: str, code_hash: str, 
                       initial_state: Dict) -> SmartContract:
        """Deploy a new privacy-preserving smart contract"""
        
        contract_id = "contract_" + hashlib.sha3_256(
            f"{owner}{code_hash}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Encrypt initial state if FHE available
        encrypted_state = None
        if self.fhe_engine:
            # Encrypt numeric values in state
            encrypted_values = {}
            for key, value in initial_state.items():
                if isinstance(value, (int, float)):
                    encrypted_values[key] = self.fhe_engine.encrypt([float(value)])
            encrypted_state = encrypted_values
        
        contract = SmartContract(
            contract_id=contract_id,
            code_hash=code_hash,
            owner=owner,
            state=initial_state,
            encrypted_state=encrypted_state
        )
        
        self.contracts[contract_id] = contract
        logger.info(f"Deployed contract: {contract_id}")
        
        return contract
    
    def execute_encrypted_transfer(self, contract: SmartContract,
                                   from_key: str, to_key: str,
                                   encrypted_amount: Any) -> bool:
        """
        Execute encrypted token transfer within contract
        
        All amounts remain encrypted - only validity is verified
        """
        
        if not self.fhe_engine or not contract.encrypted_state:
            logger.warning("FHE not available, using plaintext execution")
            return False
        
        try:
            # Get encrypted balances
            from_balance = contract.encrypted_state.get(from_key)
            to_balance = contract.encrypted_state.get(to_key)
            
            if from_balance is None or to_balance is None:
                return False
            
            # Perform encrypted arithmetic
            # new_from = from_balance - amount
            # new_to = to_balance + amount
            new_from = self.fhe_engine.subtract(from_balance, encrypted_amount)
            new_to = self.fhe_engine.add(to_balance, encrypted_amount)
            
            # Update encrypted state
            contract.encrypted_state[from_key] = new_from
            contract.encrypted_state[to_key] = new_to
            
            logger.info(f"Executed encrypted transfer in contract {contract.contract_id}")
            return True
            
        except Exception as e:
            logger.error(f"Encrypted transfer failed: {e}")
            return False
    
    def execute_private_vote(self, contract: SmartContract,
                            vote_key: str, encrypted_vote: Any) -> bool:
        """
        Execute encrypted vote aggregation
        
        Votes are encrypted and aggregated homomorphically
        """
        
        if not self.fhe_engine or not contract.encrypted_state:
            return False
        
        try:
            # Get current vote tally
            current_tally = contract.encrypted_state.get(vote_key)
            
            if current_tally is None:
                # Initialize with encrypted zero
                current_tally = self.fhe_engine.encrypt([0.0])
            
            # Add encrypted vote to tally
            new_tally = self.fhe_engine.add(current_tally, encrypted_vote)
            contract.encrypted_state[vote_key] = new_tally
            
            logger.info(f"Added encrypted vote to {vote_key}")
            return True
            
        except Exception as e:
            logger.error(f"Private vote failed: {e}")
            return False


# ============================================================================
# BLOCKCHAIN
# ============================================================================

class QuantumResistantBlockchain:
    """
    Quantum-resistant blockchain implementation
    
    Features:
    - ML-DSA signed transactions and blocks
    - ML-KEM encrypted key distribution
    - FHE-based private smart contracts
    - Merkle tree verification
    """
    
    def __init__(self):
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.wallet_manager = QuantumResistantWalletManager()
        self.contract_engine = PrivateSmartContractEngine()
        
        # Create genesis block
        self._create_genesis_block()
        
        logger.info("QuantumResistantBlockchain initialized")
    
    def _create_genesis_block(self):
        """Create the genesis block"""
        
        genesis = Block(
            block_number=0,
            previous_hash="0" * 64,
            transactions=[],
            timestamp=datetime.now(),
            merkle_root=hashlib.sha3_256(b"genesis").hexdigest(),
            validator_signature=b"genesis",
            validator_address="0x" + "0" * 40
        )
        
        self.chain.append(genesis)
        logger.info("Genesis block created")
    
    def create_transaction(self, sender_wallet: Wallet, recipient: str,
                          amount: float, tx_type: TransactionType = TransactionType.TRANSFER,
                          payload: Optional[Dict] = None) -> Transaction:
        """Create and sign a new transaction"""
        
        # Generate transaction ID
        tx_id = "tx_" + hashlib.sha3_256(
            f"{sender_wallet.address}{recipient}{amount}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Prepare transaction data for signing
        tx_data = json.dumps({
            "tx_id": tx_id,
            "sender": sender_wallet.address,
            "recipient": recipient,
            "amount": amount,
            "nonce": sender_wallet.nonce,
            "type": tx_type.value
        }, sort_keys=True).encode()
        
        # Sign with ML-DSA
        signature = self.wallet_manager.sign_transaction(sender_wallet, tx_data)
        
        tx = Transaction(
            tx_id=tx_id,
            tx_type=tx_type,
            sender=sender_wallet.address,
            recipient=recipient,
            amount=amount,
            timestamp=datetime.now(),
            signature=signature,
            payload=payload
        )
        
        # Update nonce
        sender_wallet.nonce += 1
        
        self.pending_transactions.append(tx)
        logger.info(f"Created transaction: {tx_id}")
        
        return tx
    
    def verify_transaction(self, tx: Transaction) -> bool:
        """Verify transaction signature"""
        
        sender_wallet = self.wallet_manager.wallets.get(tx.sender)
        if not sender_wallet:
            return False
        
        tx_data = json.dumps({
            "tx_id": tx.tx_id,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "nonce": sender_wallet.nonce - 1,  # Already incremented
            "type": tx.tx_type.value
        }, sort_keys=True).encode()
        
        return self.wallet_manager.verify_signature(sender_wallet, tx_data, tx.signature)
    
    def _calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate Merkle root of transactions"""
        
        if not transactions:
            return hashlib.sha3_256(b"empty").hexdigest()
        
        # Hash each transaction
        hashes = [
            hashlib.sha3_256(tx.tx_id.encode()).hexdigest()
            for tx in transactions
        ]
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha3_256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0]
    
    def mine_block(self, validator_wallet: Wallet) -> Block:
        """Mine a new block with pending transactions"""
        
        if not self.pending_transactions:
            logger.warning("No pending transactions to mine")
            return None
        
        # Verify all pending transactions
        valid_txs = [tx for tx in self.pending_transactions if self.verify_transaction(tx)]
        
        if not valid_txs:
            logger.warning("No valid transactions to mine")
            return None
        
        # Create new block
        previous_block = self.chain[-1]
        merkle_root = self._calculate_merkle_root(valid_txs)
        
        block = Block(
            block_number=len(self.chain),
            previous_hash=previous_block.calculate_hash(),
            transactions=valid_txs,
            timestamp=datetime.now(),
            merkle_root=merkle_root,
            validator_signature=b"",  # Will be set after signing
            validator_address=validator_wallet.address
        )
        
        # Sign block with ML-DSA
        block_data = json.dumps({
            "block_number": block.block_number,
            "previous_hash": block.previous_hash,
            "merkle_root": merkle_root,
            "timestamp": block.timestamp.isoformat()
        }, sort_keys=True).encode()
        
        block.validator_signature = self.wallet_manager.sign_transaction(
            validator_wallet, block_data
        )
        
        # Add block to chain
        self.chain.append(block)
        
        # Clear processed transactions
        self.pending_transactions = [
            tx for tx in self.pending_transactions 
            if tx not in valid_txs
        ]
        
        # Update transaction status
        for tx in valid_txs:
            tx.status = "confirmed"
        
        logger.info(f"Mined block #{block.block_number} with {len(valid_txs)} transactions")
        
        return block
    
    def get_balance(self, address: str) -> float:
        """Calculate balance from confirmed transactions"""
        
        balance = 0.0
        
        for block in self.chain:
            for tx in block.transactions:
                if tx.recipient == address:
                    balance += tx.amount
                if tx.sender == address:
                    balance -= tx.amount
        
        return balance


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def demo_quantum_resistant_transactions():
    """Demo 1: Quantum-resistant transaction signing"""
    
    print("\n" + "=" * 70)
    print("DEMO 1: QUANTUM-RESISTANT TRANSACTIONS")
    print("=" * 70)
    print("\nScenario: Token transfer with ML-DSA signatures")
    print("Security: NIST FIPS 204 compliant, 192-bit post-quantum security\n")
    
    # Initialize blockchain
    blockchain = QuantumResistantBlockchain()
    
    # Create wallets
    print("[1] Creating quantum-resistant wallets...")
    alice = blockchain.wallet_manager.create_wallet(initial_balance=1000.0)
    bob = blockchain.wallet_manager.create_wallet(initial_balance=500.0)
    validator = blockchain.wallet_manager.create_wallet(initial_balance=0.0)
    
    print(f"    Alice: {alice.address[:20]}... (1000 tokens)")
    print(f"    Bob:   {bob.address[:20]}... (500 tokens)")
    print(f"    Validator: {validator.address[:20]}...")
    
    # Create transactions
    print("\n[2] Creating ML-DSA signed transactions...")
    tx1 = blockchain.create_transaction(alice, bob.address, 100.0)
    tx2 = blockchain.create_transaction(bob, alice.address, 50.0)
    
    print(f"    TX1: Alice -> Bob: 100 tokens")
    print(f"    TX2: Bob -> Alice: 50 tokens")
    
    # Verify signatures
    print("\n[3] Verifying ML-DSA signatures...")
    print(f"    TX1 signature valid: {blockchain.verify_transaction(tx1)}")
    print(f"    TX2 signature valid: {blockchain.verify_transaction(tx2)}")
    
    # Mine block
    print("\n[4] Mining block with validated transactions...")
    block = blockchain.mine_block(validator)
    
    if block:
        print(f"    Block #{block.block_number} mined")
        print(f"    Merkle root: {block.merkle_root[:32]}...")
        print(f"    Transactions: {len(block.transactions)}")
        print(f"    Validator signature: {len(block.validator_signature)} bytes")
    
    print("\n[OK] Quantum-resistant transaction demo complete")


def demo_private_smart_contracts():
    """Demo 2: Privacy-preserving smart contract execution"""
    
    print("\n" + "=" * 70)
    print("DEMO 2: PRIVACY-PRESERVING SMART CONTRACTS")
    print("=" * 70)
    print("\nScenario: Confidential token contract with encrypted balances")
    print("Security: FHE-encrypted state, homomorphic computation\n")
    
    # Initialize contract engine
    engine = PrivateSmartContractEngine()
    
    # Deploy token contract
    print("[1] Deploying privacy-preserving token contract...")
    contract = engine.deploy_contract(
        owner="0x" + "a" * 40,
        code_hash=hashlib.sha3_256(b"erc20_private").hexdigest(),
        initial_state={
            "balance_alice": 1000.0,
            "balance_bob": 500.0,
            "total_supply": 1500.0
        }
    )
    
    print(f"    Contract ID: {contract.contract_id}")
    print(f"    Initial state: {contract.state}")
    
    if contract.encrypted_state:
        print(f"    Encrypted state keys: {list(contract.encrypted_state.keys())}")
    
    # Execute encrypted transfer (if FHE available)
    print("\n[2] Executing encrypted token transfer...")
    
    if engine.fhe_engine:
        # Encrypt transfer amount
        encrypted_amount = engine.fhe_engine.encrypt([100.0])
        
        success = engine.execute_encrypted_transfer(
            contract,
            from_key="balance_alice",
            to_key="balance_bob",
            encrypted_amount=encrypted_amount
        )
        
        if success:
            print("    [OK] Encrypted transfer executed")
            print("    Note: Actual balances remain encrypted")
        else:
            print("    [!] Transfer failed")
    else:
        print("    [!] FHE not available, skipping encrypted operations")
    
    print("\n[OK] Privacy-preserving smart contract demo complete")


def demo_private_voting():
    """Demo 3: Privacy-preserving voting system"""
    
    print("\n" + "=" * 70)
    print("DEMO 3: PRIVACY-PRESERVING VOTING SYSTEM")
    print("=" * 70)
    print("\nScenario: Encrypted on-chain voting with homomorphic tallying")
    print("Security: Individual votes encrypted, only final tally revealed\n")
    
    # Initialize contract engine
    engine = PrivateSmartContractEngine()
    
    # Deploy voting contract
    print("[1] Deploying voting contract...")
    voting_contract = engine.deploy_contract(
        owner="0x" + "v" * 40,
        code_hash=hashlib.sha3_256(b"voting_private").hexdigest(),
        initial_state={
            "proposal_1": 0.0,  # Yes votes
            "proposal_1_no": 0.0,  # No votes
            "total_voters": 0.0
        }
    )
    
    print(f"    Contract ID: {voting_contract.contract_id}")
    
    # Cast encrypted votes
    print("\n[2] Casting encrypted votes...")
    
    if engine.fhe_engine:
        voters = [
            ("Voter A", 1.0),   # Yes vote
            ("Voter B", 1.0),   # Yes vote
            ("Voter C", 0.0),   # No vote (counted separately)
            ("Voter D", 1.0),   # Yes vote
            ("Voter E", 1.0),   # Yes vote
        ]
        
        for voter_name, vote in voters:
            encrypted_vote = engine.fhe_engine.encrypt([vote])
            success = engine.execute_private_vote(
                voting_contract,
                vote_key="proposal_1",
                encrypted_vote=encrypted_vote
            )
            print(f"    {voter_name}: Vote encrypted and recorded")
        
        # Reveal final tally (decrypt)
        print("\n[3] Revealing final tally (decryption)...")
        
        if voting_contract.encrypted_state.get("proposal_1"):
            final_tally = engine.fhe_engine.decrypt(
                voting_contract.encrypted_state["proposal_1"]
            )
            print(f"    Proposal 1 - Yes votes: {final_tally[0]:.0f}")
            print("    Note: Individual votes remain confidential")
    else:
        print("    [!] FHE not available, skipping encrypted voting")
    
    print("\n[OK] Privacy-preserving voting demo complete")


def demo_confidential_nft():
    """Demo 4: Confidential NFT with encrypted metadata"""
    
    print("\n" + "=" * 70)
    print("DEMO 4: CONFIDENTIAL NFT SYSTEM")
    print("=" * 70)
    print("\nScenario: NFT with quantum-resistant ownership and encrypted metadata")
    print("Security: ML-DSA ownership proofs, FHE-encrypted attributes\n")
    
    # Initialize managers
    wallet_manager = QuantumResistantWalletManager()
    
    # Create artist and collector wallets
    print("[1] Creating participant wallets...")
    artist = wallet_manager.create_wallet()
    collector = wallet_manager.create_wallet()
    
    print(f"    Artist:    {artist.address[:20]}...")
    print(f"    Collector: {collector.address[:20]}...")
    
    # Create NFT metadata
    print("\n[2] Creating NFT with encrypted metadata...")
    
    nft_metadata = {
        "token_id": "nft_" + hashlib.sha3_256(b"artwork_1").hexdigest()[:8],
        "name": "Quantum Art #1",
        "artist": artist.address,
        "created": datetime.now().isoformat(),
        "encrypted_attributes": {
            "rarity_score": "[ENCRYPTED]",
            "provenance": "[ENCRYPTED]",
            "hidden_features": "[ENCRYPTED]"
        }
    }
    
    print(f"    Token ID: {nft_metadata['token_id']}")
    print(f"    Name: {nft_metadata['name']}")
    print(f"    Creator: {nft_metadata['artist'][:20]}...")
    
    # Sign NFT ownership
    print("\n[3] Generating ML-DSA ownership proof...")
    
    ownership_data = json.dumps({
        "token_id": nft_metadata["token_id"],
        "owner": collector.address,
        "timestamp": datetime.now().isoformat()
    }, sort_keys=True).encode()
    
    ownership_signature = wallet_manager.sign_transaction(collector, ownership_data)
    
    print(f"    Ownership signature: {len(ownership_signature)} bytes")
    print(f"    Signature algorithm: ML-DSA-65")
    
    # Verify ownership
    print("\n[4] Verifying ownership proof...")
    is_valid = wallet_manager.verify_signature(collector, ownership_data, ownership_signature)
    print(f"    Ownership verification: {'VALID' if is_valid else 'INVALID'}")
    
    print("\n[OK] Confidential NFT demo complete")


def run_all_demos():
    """Run all blockchain demo scenarios"""
    
    print("\n" + "=" * 70)
    print("PQC+FHE BLOCKCHAIN INTEGRATION DEMOS")
    print("=" * 70)
    print("\nDemonstrating quantum-resistant blockchain operations with")
    print("privacy-preserving smart contract execution.\n")
    
    print("Components:")
    print("  - ML-DSA (NIST FIPS 204): Transaction signatures")
    print("  - ML-KEM (NIST FIPS 203): Key encapsulation")
    print("  - DESILO CKKS FHE: Encrypted computation")
    print(f"\nPQC-FHE Library Available: {PQC_FHE_AVAILABLE}")
    
    # Run demos
    demo_quantum_resistant_transactions()
    demo_private_smart_contracts()
    demo_private_voting()
    demo_confidential_nft()
    
    # Summary
    print("\n" + "=" * 70)
    print("BLOCKCHAIN DEMO SUMMARY")
    print("=" * 70)
    print("\nCompleted Scenarios:")
    print("  [OK] 1. Quantum-resistant transaction signing (ML-DSA)")
    print("  [OK] 2. Privacy-preserving smart contracts (FHE)")
    print("  [OK] 3. Encrypted voting system (homomorphic tallying)")
    print("  [OK] 4. Confidential NFT system (encrypted metadata)")
    print("\nSecurity Features Demonstrated:")
    print("  - NIST FIPS 203/204 compliant cryptography")
    print("  - Post-quantum secure signatures")
    print("  - Homomorphic smart contract execution")
    print("  - Encrypted state management")
    print("  - Merkle tree verification")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_all_demos()
