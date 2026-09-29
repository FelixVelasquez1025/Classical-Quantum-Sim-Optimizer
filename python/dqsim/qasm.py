"""OpenQASM 2 to the shared dqsim transport, without simulator-specific optimization.

Qiskit is optional: install dqsim[qasm] to use this module. Native one-/two-qubit
standard gates are retained; custom and larger gates are expanded by definition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


class QASMImportError(ValueError):
    """A source cannot be faithfully represented by the supported transport."""


@dataclass
class ImportedCircuit:
    data: dict
    metadata: dict

    def model_dump_json(self):
        return json.dumps(self.data, separators=(',', ':'), allow_nan=False)

    @property
    def num_qubits(self):
        return sum(reg['size'] for reg in self.data['qregs'].values())

    @property
    def num_cbits(self):
        return sum(reg['size'] for reg in self.data['cregs'].values())


def _qiskit():
    try:
        import qiskit
        return qiskit
    except ImportError as exc:
        raise ImportError('OpenQASM import requires the optional dependency: pip install "dqsim[qasm]"') from exc


# Operand and parameter names match src/types.rs. Keeping the common arity at
# two means the identical transport can be passed to statevector, MPS and P-block.
_SINGLE = {name: () for name in ('id', 'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg')}
_SINGLE.update(u=('theta', 'phi', 'lam'), u3=('theta', 'phi', 'lam'), u2=('phi', 'lam'),
               u1=('lam',), p=('lam',), rx=('theta',), ry=('theta',), rz=('phi',))
_CONTROLLED = {name: () for name in ('cx', 'cy', 'cz', 'ch', 'csx')}
_CONTROLLED.update(crx=('theta',), cry=('theta',), crz=('lam',), cu1=('lam',), cp=('lam',),
                   cu3=('theta', 'phi', 'lam'), cu=('theta', 'phi', 'lam', 'gamma'))
_PAIR = {'swap': (), 'rxx': ('theta',), 'rzz': ('theta',)}


def from_qiskit(circuit, *, max_instructions=250_000) -> ImportedCircuit:
    """Lower a parsed circuit. Reject unsupported control flow and opaque gates.

    No measurements are added or removed. Global phases are not silently dropped:
    an explicit nonzero circuit/definition global phase is rejected.
    """
    _qiskit()
    from qiskit.circuit import Clbit, ClassicalRegister, IfElseOp
    if not isinstance(max_instructions, int) or max_instructions < 1:
        raise ValueError('max_instructions must be a positive integer')
    qmap = {q: i for i, q in enumerate(circuit.qubits)}
    cmap = {c: i for i, c in enumerate(circuit.clbits)}

    def registers(items, bits, mapping, kind):
        result = {}
        covered = set()
        for reg in items:
            positions = [mapping[bit] for bit in reg]
            base = positions[0] if positions else 0
            if positions != list(range(base, base + len(reg))) or covered.intersection(positions):
                raise QASMImportError(f'{kind} registers must be contiguous and disjoint')
            covered.update(positions)
            result[reg.name] = dict(name=reg.name, base=base, size=len(reg))
        if len(covered) != len(bits):
            raise QASMImportError(f'All {kind} bits must belong to a register')
        return result

    data = dict(qregs=registers(circuit.qregs, circuit.qubits, qmap, 'quantum'),
                cregs=registers(circuit.cregs, circuit.clbits, cmap, 'classical'), instructions=[])
    emitted = 0

    def lower(block, qm, cm, depth=0):
        nonlocal emitted
        if depth > 64:
            raise QASMImportError('Gate/control-flow expansion exceeds 64 levels')
        if float(block.global_phase) != 0.0:
            raise QASMImportError('Explicit circuit/definition global phase is not supported by the transport')
        output = []
        for entry in block.data:
            op = entry.operation
            qs = [qm[q] for q in entry.qubits]
            cs = [cm[c] for c in entry.clbits]
            if isinstance(op, IfElseOp):
                if len(op.blocks) != 1:
                    raise QASMImportError('Else branches are outside the OpenQASM 2 transport')
                if not isinstance(op.condition, tuple):
                    raise QASMImportError('Only register-equality conditions are supported')
                target, value = op.condition
                if isinstance(target, Clbit):
                    tested = [cm[target]]
                elif isinstance(target, ClassicalRegister):
                    tested = [cm[bit] for bit in target]
                else:
                    raise QASMImportError('Unsupported condition target')
                if not 1 <= len(tested) <= 64 or tested != list(range(tested[0], tested[0]+len(tested))):
                    raise QASMImportError('Conditions require 1..64 contiguous classical bits')
                if not 0 <= int(value) < (1 << len(tested)):
                    raise QASMImportError('Condition value exceeds register width')
                body = op.blocks[0]
                lowered = lower(body, dict(zip(body.qubits, qs)), dict(zip(body.clbits, cs)), depth+1)
                # A block condition is evaluated once. Our IR wraps individual
                # instructions, which is equivalent only if earlier writes cannot
                # change the tested value before the last instruction.
                def writes(instruction):
                    if instruction['kind'] == 'conditional': return writes(instruction['op'])
                    return instruction.get('cbit') if instruction['kind'] == 'measure' else None
                if any(writes(item) in tested for item in lowered[:-1]):
                    raise QASMImportError('Conditional block modifies its tested bits before completing')
                condition = dict(creg_base=tested[0], creg_size=len(tested), creg_value=int(value))
                output.extend(dict(kind='conditional', condition=condition.copy(), op=item) for item in lowered)
                continue
            if getattr(op, 'condition', None) is not None:
                raise QASMImportError('Unsupported legacy conditional instruction')
            name = op.name
            if name == 'measure' and len(qs) == len(cs) == 1:
                item = dict(kind='measure', qubit=qs[0], cbit=cs[0])
            elif name == 'reset' and len(qs) == 1:
                item = dict(kind='reset', qubit=qs[0])
            elif name == 'barrier':
                # The native transport uses a global no-op barrier.
                item = dict(kind='barrier')
            elif op.base_class.__module__.startswith('qiskit.circuit.library.standard_gates') and name in (_SINGLE | _CONTROLLED | _PAIR):
                if name in _SINGLE: operands, fields = ('qubit',), _SINGLE[name]
                elif name in _CONTROLLED: operands, fields = ('control', 'target'), _CONTROLLED[name]
                else: operands, fields = ('a', 'b'), _PAIR[name]
                if len(qs) != len(operands) or len(op.params) != len(fields) or cs or len(set(qs)) != len(qs):
                    raise QASMImportError(f'Invalid operands/parameters for {name}')
                values = [float(value) for value in op.params]
                if not all(math.isfinite(value) for value in values):
                    raise QASMImportError(f'Nonfinite parameter in {name}')
                item = dict(kind=name, **dict(zip(operands, qs)), **dict(zip(fields, values)))
            else:
                definition = op.definition
                if definition is None:
                    raise QASMImportError(f'Unsupported opaque operation: {name}')
                output.extend(lower(definition, dict(zip(definition.qubits, qs)),
                                    dict(zip(definition.clbits, cs)), depth+1))
                continue
            emitted += 1
            if emitted > max_instructions:
                raise QASMImportError(f'Expanded circuit exceeds {max_instructions} instructions')
            output.append(item)
        return output

    try:
        data['instructions'] = lower(circuit, qmap, cmap)
    except (TypeError, KeyError) as exc:
        raise QASMImportError(f'Unsupported circuit structure: {exc}') from exc
    return ImportedCircuit(data, {'normalization': 'common-one-two-qubit-v1',
                                 'qiskit_version': _qiskit().__version__})


_COMMENTS = re.compile(r'//[^\n]*|/\*.*?\*/', re.S)
_DECLARATIONS = re.compile(r'\b(?:gate|opaque)\s+([a-zA-Z_]\w*)')
_INCLUDES = re.compile(r'\binclude\s+"([^"\n]+)"\s*;')


def load_qasm(path, *, include_path=(), legacy_gates=True, max_source_bytes=8*1024**2,
              max_instructions=250_000) -> ImportedCircuit:
    """Read OpenQASM 2 using Qiskit's qelib1/legacy dialect and lower to dqsim.

    Explicit custom gate definitions take precedence over legacy extensions.
    Includes other than Qiskit's built-in qelib1.inc are hashed in metadata.
    """
    qiskit = _qiskit()
    from qiskit import qasm2
    if not isinstance(max_source_bytes, int) or max_source_bytes < 1:
        raise ValueError('max_source_bytes must be a positive integer')
    path = Path(path).resolve()
    search = (path.parent, *(Path(p).resolve() for p in include_path))
    declared, visited, includes = set(), set(), []
    total = 0

    def inspect(file, depth=0):
        nonlocal total
        file = file.resolve()
        if file in visited: return
        if depth > 64: raise QASMImportError('Include nesting exceeds 64 levels')
        visited.add(file)
        size = file.stat().st_size
        total += size
        if total > max_source_bytes: raise QASMImportError(f'Source/includes exceed {max_source_bytes} bytes')
        raw = file.read_bytes()
        source = _COMMENTS.sub('', raw.decode('utf-8'))
        declared.update(_DECLARATIONS.findall(source))
        if file != path:
            includes.append({'path': str(file), 'sha256': hashlib.sha256(raw).hexdigest()})
        for name in _INCLUDES.findall(source):
            if name == 'qelib1.inc': continue  # Qiskit resolves this before include_path.
            found = next((directory / name for directory in search if (directory / name).is_file()), None)
            if found is None: raise QASMImportError(f'Include not found: {name}')
            inspect(found, depth+1)
        return raw

    try:
        raw = inspect(path)
        extensions = tuple(spec for spec in qasm2.LEGACY_CUSTOM_INSTRUCTIONS if spec.name not in declared) if legacy_gates else ()
        parsed = qasm2.load(path, include_path=search, include_input_directory=None,
                           custom_instructions=extensions)
        imported = from_qiskit(parsed, max_instructions=max_instructions)
    except (qasm2.QASM2Error, UnicodeError, OSError) as exc:
        raise QASMImportError(f'{path.name}: {exc}') from exc
    imported.metadata.update(source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
                             includes=includes, legacy_gates=legacy_gates,
                             parser=f'qiskit.qasm2/{qiskit.__version__}')
    return imported
