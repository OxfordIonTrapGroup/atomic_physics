module bloch

export make_hamiltonian

using QuantumOptics

function make_hamiltonian(
    num_states::Int64,
    rabi_matrix::Matrix{ComplexF64},
    detuning_matrix::Matrix{Float64}
)
    basis = NLevelBasis(num_states)

    elements = findall(x -> x != 0., rabi_matrix)
    hamiltonians = Vector{QuantumOpticsBase.TimeDependentSum}(undef, length(elements))

    for (hamiltonian_index, state_index) in enumerate(elements)
        rabi = rabi_matrix[state_index]
        detuning = detuning_matrix[state_index]

        sigma_plus = transition(basis, state_index[1], state_index[2])

        hamiltonian_static = 1 / 2 * rabi * sigma_plus
        hamiltonian_half = TimeDependentSum((t->exp(-1im*detuning*t))=>hamiltonian_static)
        hamiltonians[hamiltonian_index] = hamiltonian_half + dagger(hamiltonian_half)
    end
    return sum(hamiltonians)
end

end
