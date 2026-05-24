## Data
* Large tick assets, level 2 limit order book from databento
* symbol: PFE, MSFT
* Vendor: NASDAQ ITCH TotalView
### Data description
#### Column

| Name | Datatype | Description |
| --- | --- | --- |
| ts_recv | Int | The capture-server(databento)-received timestamp expressed as the number of nanoseconds |
| ts_event |Int | The matching-engine-received timestamp expressed as the number of nanoseconds |
| action | Char | The event action. A for Add, C for Cancel, M for Modify, R for clear book, or T for Trade.|
| side | Char | A(Ask) for a sell order (or sell aggressor in a trade), B(Bid) for a buy order (or buy aggressor in a trade), or (N)None where no side is specified. |
| depth | Int | The book level where the update event occurred.|
| price | Int | The order price where every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001. |
| size | Int | The order quantity.|
| flags | Int | A bit field indicating event end, message characteristics, and data quality. |
| bid_px_N | Int | The bid price at level N |
| ask_px_N | Int | The ask price at level N |
| bid_sz_N | Int | The bid size at level N (top level if N = 00). |
| ask_sz_N | Int | The ask size at level N (top level if N = 00). |
### Data Process 
#### Price
For all the order price (price, bid_px_N, ask_px_N), every 1 unit corresponds to 1e-9, i.e. 1/1,000,000,000 or 0.000000001.
#### Flag
| Flag | Value | Decimal | Description |
| --- | --- | --- | --- |
| `F_LAST` | `1 << 7` | 128 | Marks the last record in a single event for a given `instrument_id`. |
| `F_TOB` | `1 << 6` | 64 | Top-of-book message, not an individual order. |
| `F_SNAPSHOT` | `1 << 5` | 32 | Message sourced from a replay, such as a snapshot server. |
| `F_MBP` | `1 << 4` | 16 | Aggregated price level message, not an individual order. |
| `F_BAD_TS_RECV` | `1 << 3` | 8 | The `ts_recv` value is inaccurate due to clock issues or packet reordering. |
| `F_MAYBE_BAD_BOOK` | `1 << 2` | 4 | An unrecoverable gap was detected in the channel. |
| `F_PUBLISHER_SPECIFIC` | `1 << 1` | 2 | Semantics depend on the `publisher_id`. Refer to the relevant dataset supplement for more details. |
|  | `1 << 0` | 1 | Reserved for internal use can safely be ignored. May be set or unset. |
### Special event process of NASDAQ-ITCH
#### Normal data:

#### Order Executed: Trade, then Cancel
* The trade session is ended with cancel with flag 130 (128+2, 2 for futher processing)
* Note that in the following session, the size traded is not update to `ask_sz_00`. It is updated in the next row.
    * The 100 traded size in row 6858123 is aggregated in row 6858124
    * Normal one like index 6858129, the add 1000 is aggregated into `bid_sz_00`
* For the data process there, focus on the size, price, and depth's changes. Validate the limit order book snapshot change at the end of this session trade with code 130. (The cancel of code 130 should not be included here. Or the queue size will be wrong)   

Table 1:

| | ts_recv | ts_event | rtype | publisher_id | instrument_id | action | side | depth | price | size | flags | ts_in_delta | sequence | bid_px_00 | ask_px_00 | bid_sz_00 | ask_sz_00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **6858122** | 1746028847553396225 | 1746028847553230136 | 10 | 2 | 12497 | C | B | 0 | 24190000000 | 300 | 130 | 166089 | 376994204 | 24190000000 | 24200000000 | 4700 | 2024 |
| **6858123** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | T | B | 0 | 24200000000 | 100 | 0 | 173211 | 377009547 | 24190000000 | 24200000000 | 4700 | 2024 |
| **6858124** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | T | B | 0 | 24200000000 | 5 | 0 | 173211 | 377009548 | 24190000000 | 24200000000 | 4700 | 1924 |
| **6858125** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | T | B | 0 | 24200000000 | 100 | 0 | 173211 | 377009549 | 24190000000 | 24200000000 | 4700 | 1919 |
| **6858126** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | T | B | 0 | 24200000000 | 14 | 0 | 173211 | 377009550 | 24190000000 | 24200000000 | 4700 | 1819 |
| **6858127** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | T | B | 0 | 24200000000 | 81 | 130 | 173211 | 377009551 | 24190000000 | 24200000000 | 4700 | 1805 |
| **6858128** | 1746028848637788498 | 1746028848637615287 | 10 | 2 | 12497 | C | A | 0 | 24200000000 | 81 | 130 | 173211 | 377009551 | 24190000000 | 24200000000 | 4700 | 1724 |
| **6858129** | 1746028848637804435 | 1746028848637638230 | 10 | 2 | 12497 | A | B | 0 | 24190000000 | 1000 | 130 | 166205 | 377009553 | 24190000000 | 24200000000 | 5700 | 1724 |

**For any trade event, the aggregated book is in the next row**

#### Side is N: 
* When the side is N, often implies an queue level is depleted or created
* In time(`ts_recv`) 1743496256564789986, two trade depleted queue level price = 25.30 and the new order is added make 25.30 as the new ask best queue. 

| | ts_recv | ts_event | rtype | publisher_id | instrument_id | action | side | depth | price | size | flags | ts_in_delta | sequence | bid_px_00 | ask_px_00 | bid_sz_00 | ask_sz_00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **947** | 1743496252424451599 | 1743496252424285113 | 10 | 2 | 12497 | C | B | 0 | 25300000000 | 90 | 130 | 166486 | 3228259 | 25300000000 | 25320000000 | 216 | 300 |
| **948** | 1743496256564960028 | 1743496256564789986 | 10 | 2 | 12497 | T | A | 0 | 25300000000 | 16 | 0 | 170042 | 3235279 | 25300000000 | 25320000000 | 216 | 300 |
| **949** | 1743496256564960028 | 1743496256564789986 | 10 | 2 | 12497 | T | A | 0 | 25300000000 | 200 | 0 | 170042 | 3235280 | 25300000000 | 25320000000 | 200 | 300 |
| **950** | 1743496256564960028 | 1743496256564789986 | 10 | 2 | 12497 | A | N | 0 | 25300000000 | 285 | 130 | 170042 | 3235281 | 25290000000 | 25300000000 | 170 | 285 |
| **951** | 1743496256564978689 | 1743496256564812482 | 10 | 2 | 12497 | A | A | 1 | 25310000000 | 72 | 130 | 166207 | 3235282 | 25290000000 | 25300000000 | 170 | 285 |

**If we found side is N, see decide the event based on the book (`ask/bid_px/sz_XX`)**

### My inference
1. Any series of trade event (in the same ts_recv) should be ended with a cancel order with flag 130. The total traded size will combined into book then (table 1)
2. If any event has the side N. 
    1. if the previous event is trade. There is a missing normal cancel (side is not N) to end the previous trade seciton (The traded size and event size of side N are combined into the book in side N row)
        1. if the side N row event is trade: The N-side row trade event is invalid. But the previous trade (idx 3604) is valid but missing a terminal cancel event

        | | ts_recv | ts_event | rtype | publisher_id | instrument_id | action | side | depth | price | size | flags | ts_in_delta | sequence | bid_px_00 | ask_px_00 | bid_sz_00 | ask_sz_00 |
        |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
        | **3603** | 1743507031815480212 | 1743507031815313973 | 10 | 2 | 12497 | C | A | 0 | 25260000000 | 200 | 130 | 166239 | 18767908 | 25250000000 | 25260000000 | 100 | 100 |
        | **3604** | 1743507031815564685 | 1743507031815393501 | 10 | 2 | 12497 | T | B | 0 | 25260000000 | 100 | 0 | 171184 | 18767909 | 25250000000 | 25260000000 | 100 | 100 |
        | **3605** | 1743507031815564685 | 1743507031815393501 | 10 | 2 | 12497 | T | N | 0 | 25260000000 | 98 | 130 | 171184 | 18767910 | 25250000000 | 25280000000 | 100 | 300 |
        | **3606** | 1743507031815872370 | 1743507031815705594 | 10 | 2 | 12497 | A | B | 0 | 25250000000 | 200 | 130 | 166776 | 18767911 | 25250000000 | 25280000000 | 300 | 300 |

        2. The N-side event is not trade. The N-side event still valid

    2. if previous event is not trade. there are multiple event in a single time \
        From index 359 -> 360, the queue size change both in bid_sz_00 and ask_sz_00

        | | ts_recv | ts_event | rtype | publisher_id | instrument_id | action | side | depth | price | size | flags | ts_in_delta | sequence | bid_px_00 | ask_px_00 | bid_sz_00 | ask_sz_00 |
        |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
        | **359** | 1743494768069998898 | 1743494768069832046 | 10 | 2 | 12497 | A | A | 1 | 25320000000 | 500 | 130 | 166852 | 969295 | 25290000000 | 25310000000 | 170 | 100 |
        | **360** | 1743494784596779460 | 1743494784596612510 | 10 | 2 | 12497 | A | N | 0 | 25310000000 | 200 | 130 | 166950 | 988058 | 25290000000 | 25310000000 | 270 | 300 |
        | **361** | 1743494785004903035 | 1743494785004735030 | 10 | 2 | 12497 | C | B | 1 | 25280000000 | 500 | 128 | 168005 | 988384 | 25290000000 | 25310000000 | 270 | 300 |
