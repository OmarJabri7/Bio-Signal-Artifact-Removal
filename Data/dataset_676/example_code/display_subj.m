% (C) 2018 Wanting Huang <172258368@qq.com>
% (C) Bernd Porr <bernd.porr@glasgow.ac.uk>
% 
% GNU GENERAL PUBLIC LICENSE
% Version 3, 29 June 2007
% 
% Demo program which plots one experiment (Temple Run) from
% subject 20. The plot shows the 3 channels recorded: Fp1,
% EMG at the chin and the switch which acts as an electronic
% clapper board.
	 
% only for octave
pkg load signal

% loads data from subject 20 and the experiment templerun
mydata = load("-ascii","../experiment_data/subj20/templerun/emgeeg.dat");

% change in case you need just some part of it
first_sample = 1;
last_sample = size(mydata)(1);

% scaling by the gain of the pre-amp
fp1 = mydata(first_sample:last_sample,2)/500;
chin = mydata(first_sample:last_sample,3)/500;
trigger = mydata(first_sample:last_sample,4);

% sampling rate
fs = 1000;

% filters
[bfilt25hz afilt25hz] = butter(2,[24/fs*2 26/fs*2],'stop');
[bfilt50hz afilt50hz] = butter(2,[49/fs*2 51/fs*2],'stop');
[bfilt80hz afilt80hz] = butter(2,[78/fs*2 82/fs*2],'stop');
[bfilt100hz afilt100hz] = butter(2,[95/fs*2 105/fs*2],'stop');
[bhp ahp] = butter(4,0.5/fs*2,'high');

fp1 = filter(bhp,ahp,filter(bfilt50hz,afilt50hz,fp1));
chin = filter(bhp,ahp,filter(bfilt50hz,afilt50hz,chin));

fp1 = filter(bfilt80hz,afilt80hz,fp1);
chin = filter(bfilt80hz,afilt80hz,chin);

fp1 = filter(bfilt100hz,afilt100hz,fp1);
chin = filter(bfilt100hz,afilt100hz,chin);

% plotting the data
t = linspace(0,last_sample/fs,last_sample);
offset = 0.001;
plot(t,fp1+offset/2,t,chin-offset/2,t,trigger/5000);
xlabel("t/sec");
ylabel("V/Fp1 & chin & trigger");
pause()
