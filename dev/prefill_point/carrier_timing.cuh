// Included inside a GENERATED copy of the original carrier's main, after
// precision, fixture, linked-backend and negative controls have passed.
json point_timings=json::array();
if(request.value("trace_only",false)) {
    nc.check(nc.point_profile(nc.session));
    proofs.push_back(proof(point.copies-1));compare_native(original,download(point,point.copies-1));
}
for(int round=0;!request.value("trace_only",false) && round<rounds;++round) for(int pass=0;pass<2;++pass) {
    const int rotate=(pass+round)%2;
    for(int mode=0;mode<3;++mode) {
        for(int stage=0;stage<(mode==0?int(inventory.size()):1);++stage) {
            const int copies=rotate?point.copies:1;
            const int width=mode==1?int(inventory.size()):1;
            std::vector<double> us(std::max(samples,copies)*width);
            nc.check(nc.point_samples(nc.session,rotate,mode,stage,samples,us.data()));
            point_timings.push_back({{"round",round},{"rotate",rotate},{"mode",mode},
                {"stage",stage},{"width",width},{"samples_us",us}});
            if(mode==2) {
                const int last=(std::max(samples,copies)-1)%copies;
                proofs.push_back(proof(last));compare_native(original,download(point,last));
            }
        }
    }
}
