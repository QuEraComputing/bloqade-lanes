# from dataclasses import dataclass, field

# RetType = TypeVar("RetType")

# # Implicit here is that the simulation backend is coupled to the choice of compilation pipeline. The question is if and how we want to decouple these two concerns.
# @dataclass
# class PhysicalSimulator:
#     noise_model: LogicalNoiseModelABC = field(default_factory=generate_simple_noise_model)

#     # Going to put the compiler-specific stuff here
#     def task(
#         self,
#         physical_kernel: Union[ir.Method[[], RetType], Callable[..., Any]],
#         m2dets: list[list[int]] | None = None,
#         m2obs: list[list[int]] | None = None,
#         arch_spec: ArchSpec = get_arch_spec(),
#         place_opt_type: type[passes.Pass] = field(default=SequentialPlacePass)
#     ) -> PhysicalSimulatorTask[RetType]:
#         physical_pipeline = PhysicalPipeline(arch_spec=arch_spec, place_opt_type=place_opt_type)
#         physical_move_kernel = physical_pipeline.emit(physical_kernel)


# @dataclass(frozen=True)
# class PhysicalSimulatorTask(Generic[RetType]):
