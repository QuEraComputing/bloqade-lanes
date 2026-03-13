from bloqade.lanes.analysis.atom import lattice


def test_unknown_and_bottom_singleton():
    assert lattice.Unknown() is lattice.Unknown()
    assert lattice.Bottom() is lattice.Bottom()
    assert lattice.Unknown().is_structurally_equal(lattice.Unknown())
    assert lattice.Bottom().is_structurally_equal(lattice.Bottom())


def test_value_structural_and_subset():
    v1 = lattice.Value(42)
    v2 = lattice.Value(42)
    v3 = lattice.Value(7)
    assert v1.is_structurally_equal(v2)
    assert v1.is_subseteq_Value(v2)
    assert not v1.is_structurally_equal(v3)
    assert not v1.is_subseteq_Value(v3)
    assert v1.copy() == v1


def test_atomstate_structural_and_subset():
    a1 = lattice.AtomState()
    a2 = lattice.AtomState()
    assert a1.is_structurally_equal(a2)
    assert a1.is_subseteq_AtomState(a2)
    assert a1.copy() == a1


def test_measurefuture_copy_and_subset():
    mf1 = lattice.MeasureFuture({})  # empty results
    mf2 = lattice.MeasureFuture({})
    assert mf1.copy() == mf1
    assert mf1.is_subseteq_MeasureFuture(mf2)


def test_measureresult_copy_and_subset():
    mr1 = lattice.MeasureResult(1)
    mr2 = lattice.MeasureResult(1)
    mr3 = lattice.MeasureResult(2)
    assert mr1.copy() == mr1
    assert mr1.is_subseteq_MeasureResult(mr2)
    assert not mr1.is_subseteq_MeasureResult(mr3)


def test_detector_and_observable_result():
    v = lattice.Value(1)
    d1 = lattice.DetectorResult(v)
    d2 = lattice.DetectorResult(v)
    assert d1.copy() == d1
    assert d1.is_subseteq_DetectorResult(d2)
    assert isinstance(d1.join_DetectorResult(d2), lattice.DetectorResult)
    assert isinstance(d1.meet_DetectorResult(d2), lattice.DetectorResult)
    o1 = lattice.ObservableResult(v)
    o2 = lattice.ObservableResult(v)
    assert o1.copy() == o1
    assert o1.is_subseteq_ObservableResult(o2)
    assert isinstance(o1.join_ObservableResult(o2), lattice.ObservableResult)
    assert isinstance(o1.meet_ObservableResult(o2), lattice.ObservableResult)


def test_ilist_and_tuple_result():
    v = lattice.Value(1)
    il1 = lattice.IListResult((v, v))
    il2 = lattice.IListResult((v, v))
    il3 = lattice.IListResult((v,))
    assert il1.copy() == il1
    assert il1.is_subseteq_IListResult(il2)
    assert not il1.is_subseteq_IListResult(il3)
    assert isinstance(il1.join_IListResult(il2), lattice.IListResult)
    assert isinstance(il1.meet_IListResult(il2), lattice.IListResult)
    assert isinstance(il1.join_IListResult(il3), lattice.Unknown)
    assert isinstance(il1.meet_IListResult(il3), lattice.Bottom)

    t1 = lattice.TupleResult((v, v))
    t2 = lattice.TupleResult((v, v))
    t3 = lattice.TupleResult((v,))
    assert t1.copy() == t1
    assert t1.is_subseteq_TupleResult(t2)
    assert not t1.is_subseteq_TupleResult(t3)
    assert isinstance(t1.join_TupleResult(t2), lattice.TupleResult)
    assert isinstance(t1.meet_TupleResult(t2), lattice.TupleResult)
    assert isinstance(t1.join_TupleResult(t3), lattice.Unknown)
    assert isinstance(t1.meet_TupleResult(t3), lattice.Bottom)
