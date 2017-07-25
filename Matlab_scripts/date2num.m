function date = date2num(dates)
    data_size = length(dates);
    year = zeros(data_size, 1);
    month = zeros(data_size, 1);
    day = zeros(data_size, 1);
    for i = 1 : data_size
        split = strsplit(char(dates(i)),'/');
        year(i) = str2double(split(3));
        month(i) = str2double(split(2));
        day(i) = str2double(split(1));
    end
    date = struct('year', year, 'month', month, 'day', day);
end