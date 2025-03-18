using PyCall
using TensorQEC
using Printf
using Base.Threads
stim = PyCall.pyimport("stim")
pymatching = PyCall.pyimport("pymatching")
ldpc = PyCall.pyimport("ldpc")


for dd in 3:2:7
    for error_rate in 0.007:0.0005:0.01
        distance, rounds = dd, dd
        # error_rate = 0.01
        seed = 0
        noisy_circuit = stim.Circuit().generated(
                                        code_task="surface_code:rotated_memory_x", # "repetition_code:memory", # 
                                        distance=distance,
                                        rounds=rounds,
                                        after_clifford_depolarization=error_rate,
                                        before_measure_flip_probability=error_rate,
                                        after_reset_flip_probability=error_rate,
                                        # before_round_data_depolarization=error_rate,
                                        )
        dem_flatten = noisy_circuit.detector_error_model(flatten_loops=true)
        n_d, n_e = dem_flatten.num_detectors, dem_flatten.num_errors
        error_rates = zeros(n_e)
        stabilizer = zeros(n_d, n_e)
        logical_locs = []
        dem_str = split(dem_flatten.__str__(), "\n")
        for i in 1:n_e
            s = split(dem_str[i])
            if startswith(s[1], "error(")
                num = parse(Float64, s[1][7:end-1])
                error_rates[i] = num
            end
            for j in 2:length(s)
                if startswith(s[j], "D")
                    row = parse(Int, s[j][2:end])
                    stabilizer[row+1, i] = 1
                end
                if startswith(s[j], "L") && !(i in logical_locs)
                    push!(logical_locs, i)
                end
            end
        end

        num_samples = 100000
        samples, logical_samples = noisy_circuit.compile_detector_sampler(
            seed=seed).sample(num_samples, separate_observables=true)

        error_instances = []

        decoder = IPDecoder()

        model = noisy_circuit.detector_error_model(decompose_errors=true)
        matching = pymatching.Matching.from_detector_error_model(model)

        t0 = time()
        predicted_observables_matching = convert(Matrix{Bool}, matching.decode_batch(samples))
        t1 = time()
        error_instances_matching = findall(predicted_observables_matching .!= logical_samples)
        @printf("distance: %d error rate: %.4f time: %.4f logical error rate: %.5f\n", distance, error_rate, t1-t0, length(error_instances_matching)/num_samples)

        bposd = ldpc.bposd_decoder(stabilizer, channel_probs=error_rates)
        error_instances_bposd = []
        t0 = time()
        @threads for i in 1:num_samples # segment default
            res = bposd.decode(samples[i, :])
            # res = bposd.osdw_decoding
            success = sum(res[logical_locs]) % 2 == logical_samples[i]
            if !success
                push!(error_instances_bposd, i)
                # println(i, " ", samples[i, :], ' ', res.error_qubits[logical_locs], ' ', samples[i])
            end
        end
        t1 = time()
        @printf("distance: %d error rate: %.4f time: %.4f logical error rate: %.5f\n", distance, error_rate, t1-t0, length(error_instances_bposd)/num_samples)

        t0 = time()
        @threads for i in 1:num_samples
            res = decode(decoder, convert(Matrix{Bool}, stabilizer), samples[i, :], error_rates)
            success = sum(res.error_qubits[logical_locs]).x == logical_samples[i]
            if !success
                push!(error_instances, i)
                # println(i, " ", samples[i, :], ' ', res.error_qubits[logical_locs], ' ', samples[i])
            end
        end
        t1 = time()
        @printf("distance: %d error rate: %.4f time: %.4f logical error rate: %.5f\n", distance, error_rate, t1-t0, length(error_instances)/num_samples)
    end
end